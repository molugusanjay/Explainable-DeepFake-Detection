# Run the project locally in VS Code

Open `DeepFake_Detection_FYP` using File > Open Folder. Run commands from this folder. The trained checkpoint and YuNet detector are included; the training datasets are separate. A paid cloud account is not required for local prediction.

## Existing laptop

Double-click `Start_App.bat`. It changes to the project folder, checks the checkpoint, and looks for Python in `.venv`, `venv`, then `..\work\venv`. Keep the terminal open while using the browser. Close the server with Ctrl+C.

For a VS Code terminal using PowerShell:

```powershell
& '..\work\venv\Scripts\python.exe' src/app.py --open-browser
```

Select `C:\Users\heman\Documents\DeepFake_Project\work\venv\Scripts\python.exe` in VS Code's Python interpreter selector on this laptop. This location is specific to the current arrangement. The reorganized package does not include a configured VS Code workspace or tasks file.

The default browser address is `http://127.0.0.1:7860`. The first prediction loads the model. Upload an original photograph with one clear face, click **Analyze image**, then inspect the prediction, confidence, original image, heatmap and explanation. The guidebook lists saved REAL, FAKE and error examples. Use original images, not their colored overlays, as uploads.

## Another computer

Create a Python 3.12 environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/run_local.py check
.\.venv\Scripts\python.exe src/app.py --open-browser
```

`ENVIRONMENT_LOCK.txt` records the original installed versions. `requirements.txt` allows version ranges, so a fresh installation is not a guarantee of bitwise reproduction. Use a PyTorch build suitable for the computer's hardware if GPU support is required. The runtime also supports CPU prediction, with different performance. Installation needs internet access; prediction uses local model files.

## Troubleshooting

- Missing checkpoint: restore `models/checkpoints/final_frequency/calibrated.pt`.
- Missing face detector: restore `models/assets/face_detection_yunet_2023mar.onnx`.
- Import error: select the intended environment and install requirements there.
- Port occupied: launch `python src/app.py --port 7861 --open-browser` and open that port.
- Rejected image: use one sufficiently large, visible face. A rejected upload is not a REAL prediction.
- Blank heatmap: a zero positive Grad-CAM map is allowed. Do not add artificial marks.

Use `docs/REPRODUCE.md` for experiment prerequisites. Do not start training merely to demonstrate the supplied checkpoint.
