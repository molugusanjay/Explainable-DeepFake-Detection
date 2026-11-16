"""
DeepFake Detection - Gradio Web Application
============================================
Final Year Project: CNN + Frequency-Based Explainable DeepFake Detection

Launch locally:
    python src/app.py --checkpoint models/checkpoints/final_frequency/calibrated.pt

Or double-click Start_App.bat on Windows.
"""
import argparse
import os
from pathlib import Path

# Resolve project root (two levels up from src/app.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
os.environ.setdefault('GRADIO_TEMP_DIR',
    str(_PROJECT_ROOT / 'models' / 'assets' / 'gradio_temp'))
os.environ.setdefault('MPLCONFIGDIR',
    str(_PROJECT_ROOT / 'models' / 'assets' / 'matplotlib'))
# Load the model libraries before the UI to avoid slow Windows source inspection
# over Gradio's large import graph during PyTorch operator registration.
from deepfake.predict import Predictor
from deepfake.ai_detector import get_ai_classifier
import gradio as gr

# Pre-load the AI detector into memory so the first image upload is completely instant
print("Pre-loading Visual AI Detector into memory...", flush=True)
get_ai_classifier()
print("Pre-loading finished!", flush=True)

def build_app(checkpoint='models/checkpoints/final_frequency/calibrated.pt'):
    state = {'predictor': None}
    ready = Path(checkpoint).is_file()

    from deepfake.metadata import analyze_metadata
    from deepfake.ai_detector import detect_ai_generation
    
    def analyze(image):
        try:
            if state['predictor'] is None:
                state['predictor'] = Predictor(checkpoint)

            # Perform background metadata check (returns info string + camera detected flag)
            meta_info, has_real_camera = analyze_metadata(image)

            # Predict with CNN
            result = state['predictor'].predict(image)

            # FAST AI Generator Check
            ai_info = detect_ai_generation(image)

            # Override to FAKE only when there is strong evidence AND no real camera was found.
            # If a real camera Make/Model is present in EXIF, we trust it is a genuine photo
            # and do NOT override — this prevents false positives on real phone photographs.
            is_ai_visual = ai_info and "STRONG VISUAL AI SIGNATURE DETECTED" in ai_info
            is_ai_meta = "STRONG AI SIGNATURE DETECTED" in meta_info

            if (is_ai_visual or is_ai_meta) and not has_real_camera:
                result['prediction'] = 'FAKE (AI Generated)'
                result['confidence'] = 0.99
                reason = ai_info if is_ai_visual else "Metadata AI signature found (no camera data present)."
                result['explanation'] = "Pure AI Generation detected. " + reason + "\n\n" + result['explanation']
                if is_ai_visual:
                    meta_info = meta_info + "\n\n⚠️ AI DETECTOR: " + ai_info

            note = ('Red highlights support the FAKE prediction.' if 'FAKE' in result['prediction']
                    else 'Blue highlights support the REAL prediction; they do not mark suspected edits.')
            summary = {k: v for k, v in result.items() if k not in ('original', 'overlay')}

            return (meta_info, result['prediction'], f"{result['confidence']:.1%}",
                    result['original'], result['overlay'],
                    f"{result['explanation']}\n\n{note}\n\n{result['confidence_type']} model confidence.", summary)
        except (ValueError, FileNotFoundError) as error:
            # Explicitly clear old results after an invalid upload.
            return 'Failed to analyze', 'Not analyzed', '', image, None, str(error), {}


    with gr.Blocks(title='Face Evidence | DeepFake Detection') as demo:
        gr.Markdown('# Face Evidence\n### CNN + frequency-based explainable DeepFake detection\n'
                    'Upload a photograph containing one clear face to inspect the model prediction and its visual evidence.')
        if not ready:
            gr.Markdown('**Training required.** The application is ready, but no trained checkpoint is installed. '
                        'Complete local training and calibration before using detection.')
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(type='filepath', label='Upload a facial photograph', sources=['upload'])
                button = gr.Button('Analyze image', variant='primary')
                gr.Markdown('This model is intended for research on facial manipulation. '
                            'Performance on new generators, screenshots and heavily compressed images is unverified.')
            with gr.Column(scale=2):
                metadata_check = gr.Textbox(label='0 · Metadata Background Check', lines=3, interactive=False)
                with gr.Row():
                    label = gr.Textbox(label='1 · Prediction', interactive=False)
                    confidence = gr.Textbox(label='2 · Model confidence', interactive=False)
                with gr.Row():
                    original = gr.Image(label='Original photograph', interactive=False)
                    overlay = gr.Image(label='3 · Facial Grad-CAM overlay', interactive=False)
                explanation = gr.Textbox(label='4 · Explanation', lines=6, interactive=False)
                with gr.Accordion('Analysis details', open=False):
                    details = gr.JSON(label='Result metadata')
        button.click(analyze, upload, [metadata_check, label, confidence, original, overlay, explanation, details], concurrency_limit=1)
        gr.Markdown('Heatmaps show regions associated with the prediction. They are not verified manipulation boundaries. '
                    'The display will now analyze the entire image or bounding box dynamically.')
    return demo


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='models/checkpoints/final_frequency/calibrated.pt')
    p.add_argument('--port', type=int, default=7860)
    p.add_argument('--open-browser', action='store_true')
    args = p.parse_args()
    build_app(args.checkpoint).launch(server_name='127.0.0.1', server_port=args.port, share=False,
                                     inbrowser=args.open_browser)
