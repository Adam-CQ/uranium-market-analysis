"""Execute every workshop notebook in a clean kernel; keep outputs local."""
from datetime import datetime, timezone
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent.parent
    output = root / 'reports/runs' / datetime.now(timezone.utc).strftime('notebook_checks_%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    summary = []
    for path in sorted((root / 'notebooks').glob('0[1-7]_*.ipynb')):
        print(f'Executing {path.name}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        try:
            NotebookClient(notebook, timeout=600, kernel_name='python3',
                           resources={'metadata': {'path': str(root)}}).execute()
            summary.append({'notebook': path.name, 'status': 'passed',
                            'code_cells': sum(c.cell_type == 'code' for c in notebook.cells)})
        except Exception as error:
            summary.append({'notebook': path.name, 'status': 'failed', 'error': str(error)})
            (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            raise
        finally:
            nbformat.write(notebook, output / path.name)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'All {len(summary)} notebooks passed. Evidence: {output}', flush=True)


if __name__ == '__main__':
    main()
