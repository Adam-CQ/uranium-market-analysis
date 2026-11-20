"""Annual expanding fits, dated predictions, training-only preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_COLUMNS


def estimators(cfg):
    return {
        'majority': DummyClassifier(strategy='prior'),
        'logistic': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0, random_state=cfg['seed'])),
        'random_forest': make_pipeline(SimpleImputer(strategy='median'),
                                      RandomForestClassifier(n_estimators=100, max_depth=4,
                                                             min_samples_leaf=20, n_jobs=1,
                                                             random_state=cfg['seed'])),
    }


def probability(model, features):
    classes = model.classes_
    if 1 not in classes:
        return np.zeros(len(features))
    return model.predict_proba(features)[:, list(classes).index(1)]


def walk_forward(table, cfg):
    eligible = table[table.eligible & table[FEATURE_COLUMNS].notna().any(axis=1)].copy()
    candidates = eligible[eligible.date >= cfg['validation_start']]
    predictions, audits = [], []
    for year, test in candidates.groupby(candidates.date.dt.year):
        cutoff = test.date.min()
        train = eligible[(eligible.date < cutoff) & (eligible.label_end < cutoff) & eligible.outperform_1m.notna()]
        if train.date.nunique() < cfg['min_train_months'] or train.outperform_1m.nunique() < 2:
            continue
        # A full calendar-year prediction block uses one fit at the first signal date.
        for name, template in estimators(cfg).items():
            model = clone(template).fit(train[FEATURE_COLUMNS], train.outperform_1m.astype(int))
            out = test[['date', 'ticker', 'label_end', 'outperform_1m', 'forward_return',
                        'benchmark_forward_return', 'trade_return', 'execution_date', 'exit_date']].copy()
            out['model'] = name
            out['probability'] = probability(model, test[FEATURE_COLUMNS])
            out['fit_cutoff'] = cutoff
            predictions.append(out)
        momentum = predictions[-1].copy()
        momentum['model'] = 'momentum'
        momentum['probability'] = (test.rel_63 > 0).astype(float).to_numpy()
        predictions.append(momentum)
        audits.append({'year': year, 'fit_cutoff': cutoff, 'train_rows': len(train),
                       'train_months': train.date.nunique(), 'latest_training_label_end': train.label_end.max()})
    if not predictions:
        raise ValueError('Insufficient history for chronological evaluation')
    result = pd.concat(predictions, ignore_index=True)
    result['split'] = np.where(result.date < pd.Timestamp(cfg['holdout_start']), 'validation', 'holdout')
    return result, pd.DataFrame(audits)


def classification_metrics(predictions):
    records, calibration = [], []
    known = predictions.dropna(subset=['outperform_1m'])
    for (name, split), g in known.groupby(['model', 'split']):
        y = g.outperform_1m.astype(int)
        p = g.probability
        predicted = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        records.append({'model': name, 'split': split, 'rows': len(g), 'months': g.date.nunique(),
                        'base_rate': y.mean(), 'balanced_accuracy': balanced_accuracy_score(y, predicted),
                        'precision': precision_score(y, predicted, zero_division=0),
                        'recall': recall_score(y, predicted, zero_division=0),
                        'roc_auc': roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
                        'pr_auc': average_precision_score(y, p) if y.nunique() == 2 else np.nan,
                        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
        bins = pd.cut(p, [0, .2, .4, .6, .8, 1], include_lowest=True)
        for label, b in g.groupby(bins, observed=True):
            calibration.append({'model': name, 'split': split, 'bin': str(label), 'count': len(b),
                                'mean_probability': b.probability.mean(), 'observed_frequency': b.outperform_1m.mean()})
    return pd.DataFrame(records), pd.DataFrame(calibration)


def select_model(metrics):
    validation = metrics[(metrics.split == 'validation') & metrics.model.isin(['logistic', 'random_forest'])]
    if validation.empty:
        raise ValueError('No validation period available for model selection')
    return validation.sort_values(['balanced_accuracy', 'model'], ascending=[False, True]).iloc[0].model


def latest_signals(table, cfg, name):
    date = table.date.max()
    train = table[table.eligible & (table.label_end <= date) & table.outperform_1m.notna()]
    latest = table[(table.date == date) & table.eligible].copy()
    if latest.empty or train.date.nunique() < cfg['min_train_months']:
        raise ValueError('Insufficient history or no eligible securities for latest signal')
    model = estimators(cfg)[name].fit(train[FEATURE_COLUMNS], train.outperform_1m.astype(int))
    latest['probability'] = probability(model, latest[FEATURE_COLUMNS])
    latest = latest.sort_values(['probability', 'ticker'], ascending=[False, True])
    selected = latest[latest.probability >= cfg['probability_threshold']].head(cfg['top_n']).ticker
    latest['selected'] = latest.ticker.isin(selected)
    latest['target_weight'] = latest.selected.astype(float) / max(len(selected), 1)
    return latest[['date', 'ticker', 'probability', 'selected', 'target_weight']], model
