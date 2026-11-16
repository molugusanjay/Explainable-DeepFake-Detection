"""Test the local Gradio upload path and close the temporary server afterwards."""
import json
import sys
from pathlib import Path

# Add src to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / 'src'))

from app import build_app
from gradio_client import Client, handle_file
from PIL import Image


def main():
    run = root / 'models' / 'checkpoints' / 'final_frequency'
    examples = json.loads((run / 'demo' / 'verification.json').read_text())['examples']
    demo = build_app(str(run / 'calibrated.pt'))
    demo.launch(server_name='127.0.0.1', server_port=7861, share=False, prevent_thread_lock=True)
    try:
        client = Client('http://127.0.0.1:7861', verbose=False, httpx_kwargs={'timeout': 30})
        checked = []
        for label in ['REAL', 'FAKE']:
            record = next(r for r in examples if r['prediction'] == label)
            filename = run / 'demo' / f"{record['number']}_original.png"
            result = client.submit(handle_file(str(filename)), api_name='/analyze').result(timeout=120)
            assert len(result) == 6 and result[0] == label and result[3]
            assert result[5]['confidence_type'] == 'Validation-calibrated'
            assert result[5]['heatmap_class'] == label
            checked.append(dict(prediction=result[0], confidence=result[1], heatmap_returned=True))
        blank = root / 'models' / 'assets' / 'blank_upload_check.png'
        Image.new('RGB', (256, 256), 'white').save(blank)
        invalid = client.submit(handle_file(str(blank)), api_name='/analyze').result(timeout=60)
        assert invalid[0] == 'Not analyzed' and invalid[3] is None and invalid[5] == {}
        report = dict(local_upload_api=True, prediction=result[0], confidence=result[1],
            valid_examples=checked,
            heatmap_returned=True, blank_rejected=True, stale_results_cleared=True,
            confidence_type=result[5]['confidence_type'], visual_browser_check=False)
        (run / 'app_verification.json').write_text(json.dumps(report, indent=2))
        print(report, flush=True)
    finally:
        demo.close()


if __name__ == '__main__':
    main()
