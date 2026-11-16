# DeepFake & AI Generation Detection

> **FINAL PROJECT UPDATE:** This project has been upgraded to detect modern generative AI images (like Midjourney, DALL-E, and Gemini) in addition to traditional deepfakes. 

##  How to Run the Project 

### Option 1: Automatic Setup (Recommended for Windows)
Simply double-click **`Start_App.bat`**. 
- On first launch, it will automatically create the virtual environment, install all required dependencies, and open the web application at `http://127.0.0.1:7860`.
- On subsequent launches, it will start the application immediately.

### Option 2: Manual Setup via PowerShell / Terminal
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python src/app.py --open-browser
```
The web interface will open automatically in your browser at `http://127.0.0.1:7860`.

---

##  Key Features in this Final Version
1. **Explainable CNN Model:** A custom ConvNeXt-Tiny model equipped with a Frequency Gating Mechanism (FGEM). Optimized for CPU and GPU execution with **calibrated probability metrics**.
2. **Metadata Background Check:** Automatically inspects the raw binary headers and EXIF data of uploaded images to detect hidden AI software signatures (e.g., SynthID, C2PA, Midjourney, DALL-E) or missing camera hardware metadata, while preserving authentic phone camera photos.
3. **Visual AI Detection:** Integrates HuggingFace image classification to visually scan full images for pure generative AI artifacts.
4. **Grad-CAM Explanations:** Generates heatmap overlays across the entire image bounding box to visually explain which spatial regions the CNN model found suspicious.
5. **Sample Images Provided:** Ready-to-test sample images are included in the `Test_Images/` directory for immediate testing (both REAL and FAKE examples).

---

# CNN and Frequency-Based Explainable AI Framework for DeepFake Detection

A local research prototype for classifying a **single facial image** as REAL or FAKE and displaying its estimated confidence, facial Grad-CAM overlay and spatial explanation. The implementation is inspired by MH-FFNet; it is an independent adaptation, not the paper authors' official implementation or an exact reproduction.

The packaged model and face detector support local inference. **The 95% accuracy goal was not achieved:** the selected frequency model obtained 92.66% accuracy on the combined legacy test and 89.24% on the additional DeepFakeFace holdout. These results describe accepted face crops from the evaluated subsets, not arbitrary internet images.

## Academic materials

The reviewed report, presentation, viva guide and demonstration guidebook are collected in [docs/Academic_Submission](docs/Academic_Submission/). Read their evidence and limitations alongside the measured results below. Earlier documentation may refer to the original experiment workspace; the paths in this README describe this reorganized project.

## Quick Start (For Friends & Beginners)

If you just received this project and want to run it quickly, follow these steps:

1. **Prerequisites**: Ensure you have [Python](https://www.python.org/downloads/) installed on your computer (Python 3.12 is recommended).
2. **Setup**:
   - Open a terminal (Command Prompt or PowerShell) in this project folder.
   - Run the following commands to set up the environment and install dependencies:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     pip install -r requirements.txt
     pip install -e .
     ```
3. **Run the App**: 
   - Double-click the **`Start_App.bat`** file in this folder.
   - A terminal window will open, and shortly after, a webpage will automatically open in your browser (`http://127.0.0.1:7860`) showing the DeepFake Detection interface.
   - **Tip:** To stop the application, just close the terminal window or press `Ctrl+C` in it.

## Run the application

### Existing Windows installation

Open this project folder and double-click **[Start_App.bat](Start_App.bat)**. The launcher looks for `venv\Scripts\python.exe`, then the sibling environment `..\work\venv\Scripts\python.exe`, then `python` on PATH. It starts the local application at `http://127.0.0.1:7860` and requests that your browser open it. Keep its terminal open while using the app; press Ctrl+C there to stop it.

The required packaged files are:

- `models/checkpoints/final_frequency/calibrated.pt` â€” trained classifier and temperature.
- `models/assets/face_detection_yunet_2023mar.onnx` â€” face detector.

### New Python environment

Run these commands from the project root. Python 3.12 was used in the recorded environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python src/app.py --checkpoint models/checkpoints/final_frequency/calibrated.pt --open-browser
```

Package installation requires internet access. The supplied weights allow inference without downloading a training dataset. PyTorch uses CUDA when available and otherwise runs on the CPU; CPU inference can be slower. `requirements.txt` specifies compatible ranges. `ENVIRONMENT_LOCK.txt` records the exact original environment, including torch 2.11.0+cu126, torchvision 0.26.0+cu126 and Gradio 6.29.1; it is not a guarantee that every package/platform combination reproduces identical outputs.

### Demonstration

1. Upload a clear photograph containing exactly one sufficiently large face.
2. Run the analysis and read the predicted class and confidence.
3. Compare the original photograph with the class-specific facial heatmap.
4. Read the spatial explanation and, when needed, the prediction metadata.
5. Demonstrate an invalid image too: missing, multiple or very small faces are rejected and previous outputs are cleared.

Verified example inputs and their recorded outcomes are available in `models/checkpoints/final_frequency/demo/verification.json` and the numbered `*_original.png` files beside it. This includes successful cases and errors. A provided image is not a promise of correct classification. The `Test_Images` directory provides additional local demonstration inputs, not an independent evaluation benchmark.

### Command-line inference

```powershell
python src/predict.py models/checkpoints/final_frequency/demo/1_original.png
```

The command prints prediction metadata and writes `original.png`, `heatmap.png` and `prediction.json` to `results/predictions` by default. Use `--output` to choose another directory; repeated use of the same output directory replaces these inference output files.

## Implemented architecture

The CNN runs **sequentially**. Spatial and frequency features interact inside the enhancement modules; the model does not run two independent full ConvNeXt networks in parallel.

```text
Input image
    |
YuNet: require one valid face, crop with margin
    |
Resize to 224 x 224 and ImageNet RGB normalization
    |
ConvNeXt stem + stage 1              -> 96 x 56 x 56
    |
Patch FGEM: 7 x 7 feature patches, DCT filtering, channel fusion
    |
Downsampling + ConvNeXt stage 2      -> 192 x 28 x 28
    |
Global FGEM: whole-feature-channel DCT filtering, channel fusion
    |
Remaining ConvNeXt stages 3 and 4    -> 768 x 7 x 7
    |
Global average pooling + two-output classifier
    |
REAL / FAKE logits
    |----------------------------------|
Softmax(logits / T)                    Predicted-class Grad-CAM
T = 2.7164392471                       Final stage activations
    |                                  |
Class + estimated confidence          Facial overlay + spatial text
```

The backbone has four ConvNeXt stages. In torchvision these are stored with the stem/downsampling layers in eight feature containers, indexed 0â€“7. The selected model uses ConvNeXt-Tiny and has 28,007,138 parameters.

### Frequency processing

`src/deepfake/model.py` applies a differentiable orthonormal two-dimensional DCT-II to learned feature maps, removes a low-frequency band, then reconstructs a residual with the inverse transform. The retained coefficient condition is:

```text
(u / max(H - 1, 1) + v / max(W - 1, 1)) / 2 > 0.25
```

Patch FGEM processes nonoverlapping 7 x 7 feature patches. Global FGEM processes an entire intermediate feature channel. Each module projects the spatial features into queries and the frequency residual into keys and values, computes channel attention, adds the fused result to the spatial features and applies a learned 1 x 1 projection. DCT and attention calculations use float32 internally for numerical stability. The CNN-only ablation replaces both enhancement modules with identity operations.

### Confidence and explanation

The fixed class order is REAL=0 and FAKE=1. Temperature scaling selects a positive scalar using validation cross-entropy. The stored temperature **2.7164392471** is distinct from the **0.5 probability decision threshold**. It changes probability sharpness without changing the two-logit argmax. Confidence is a model estimate; calibration on validation data does not guarantee accuracy or reliable confidence on a new distribution.

Grad-CAM differentiates the predicted-class logit through the trained model and uses the final convolutional activations. The coarse 7 x 7 map is resized into the detected face coordinates. **Red indicates positive FAKE-class support; blue indicates positive REAL-class support.** A zero map stays zero. An approximate facial oval limits the displayed overlay; the classifier still uses the full crop, and the display restriction can hide attention outside the oval.

The text names broad regions with strong displayed attention. It does not independently diagnose texture defects, blending errors, eye reflections or any particular manipulation. **Grad-CAM is not an edited-pixel segmentation mask**, and a plausible heatmap can accompany an incorrect prediction.

## Measured results and their scope

| Evaluation | Accuracy | Balanced accuracy | ROC AUC | Face images |
|---|---:|---:|---:|---:|
| CNN-only baseline, combined legacy test | 93.90% | 90.09% | 0.9725 | 2,901 |
| Selected CNN + frequency, combined legacy test | 92.66% | 89.18% | 0.9646 | 2,901 |
| Selected model, DFF part of legacy test | 89.51% | 90.70% | 0.9757 | 629 |
| Selected model, FF++ part of legacy test | 93.53% | 88.56% | 0.9680 | 2,272 |
| Selected model, additional DFF holdout | 89.24% | 91.04% | 0.9792 | 1,320 |

The selected model's per-image CSV files reproduce its saved accuracy, balanced accuracy, AUC, confusion matrices and source-group bootstrap intervals. The CNN-only row is retained in `results/runs/mixed_cnn_test_metrics.json`; its checkpoint, prediction CSV and full training history are **not included in this folder**, so that row cannot be independently re-evaluated from the package alone.

The frequency model's combined accuracy is 1.24 percentage points below the recorded CNN-only baseline; balanced accuracy is 0.91 points below. **This experiment does not demonstrate an accuracy improvement from the frequency modules.** It is a single-seed comparison with a limited training budget. The deployed model retains the CNN-plus-frequency architecture required by the project and was selected among frequency candidates using validation results.

The combined confusion matrix is `[[490, 98], [115, 2198]]`: 98 real images were flagged fake and 115 fake images were missed. Rows represent true REAL/FAKE; columns represent predicted REAL/FAKE. The additional holdout matrix is `[[282, 17], [125, 896]]`.

The 95% source-group bootstrap interval for **accuracy** is 90.85%â€“94.24% on the combined test and 87.59%â€“90.85% on the additional holdout. These are uncertainty estimates within the sampled data scope, not universal performance guarantees.

### Evaluation disclosures

- Legacy test results had already been inspected before the mixed experiment. They are repeated benchmark evaluations, not an untouched final test.
- The additional holdout uses unused DeepFakeFace source pairs from known methods. It is not an unseen-generator or entirely external-dataset benchmark. Its selection policy is recorded in the metrics, but its full frozen selection manifest is absent from this folder.
- Source grouping and exact-hash checks reduce leakage. They do not establish identity-disjoint splits or eliminate all near duplicates. Repeated identities may remain.
- Classification metrics exclude face-detection and crop-policy rejections. They describe accepted faces rather than the entire input collection.
- Validation data was reused for checkpoint selection and temperature fitting, so validation results can be optimistic.
- Neither test reaches the 95% point-estimate accuracy goal. No pixel-level localization score, multi-seed study or arbitrary-image reliability claim is supported.

## Training record and data availability

The selected checkpoint records ImageNet ConvNeXt-Tiny initialization, seed 42, AdamW learning rate 4e-5, batch size 8 with accumulation 2, zero loader workers, a six-epoch maximum and patience 3. Training uses cross-entropy, weight decay .01, CUDA mixed precision and gradient norm clipping at 1. The selected checkpoint is epoch 4, with 2,992 optimizer steps.

Mixed training sampling gives equal mass to each dataset and each class, then equalizes fake methods within each dataset. Checkpoint selection averages per-dataset validation balanced accuracy. The saved selection record retains `mixed_frequency` (92.94% validation selection score) over a transfer-initialized candidate (92.19%). The latter candidate's historical parent checkpoints and complete experiment evidence are absent here.

| Documented prepared split | Train | Validation | Test |
|---|---:|---:|---:|
| DeepFakeFace | 5,153 | 587 | 629 |
| FaceForensics++ C23 | 6,813 | 2,186 | 2,272 |
| Combined | 11,966 | 2,773 | 2,901 |

Test counts are supported by saved predictions, and validation totals by checkpoint/calibration records. Training counts come from the retained experiment documentation. `data/raw` and `data/prepared` do not contain the full datasets or manifests in this package. The source preparation, training and evaluation scripts are included, but reproducing training requires obtaining data, creating the documented manifests and preserving the split protocol. FaceForensics++ requires access from its authors; the approved private downloader is not redistributed.

**The YAML files in `configs/` are reference records only. No application or training script parses them.** Actual runs are controlled by CLI arguments and the values in the experiment runners. `train_config.yaml` describes generic standalone CLI defaults (20 epochs, workers 2, patience 5); `experiment_config.yaml` describes the executed mixed protocol (6 epochs, workers 0, patience 3). Editing YAML alone changes no behavior.

## Project map

| Path | Responsibility |
|---|---|
| `src/app.py` | Local Gradio upload interface |
| `src/predict.py` | Single-image command-line inference |
| `src/deepfake/model.py` | ConvNeXt, DCT filtering and FGEM fusion |
| `src/deepfake/faces.py` | YuNet detection, single-face checks and bounding-box crop |
| `src/deepfake/predict.py` | Prediction, calibration and explanation pipeline |
| `src/deepfake/explain.py` | Grad-CAM, overlay and region descriptions |
| `src/deepfake/data.py` | CSV manifest validation and image transforms |
| `src/deepfake/sampling.py` | Dataset/class/method balancing and validation selection |
| `src/deepfake/metrics.py` | Classification, calibration and uncertainty metrics |
| `src/train.py`, `src/evaluate.py` | Training, calibration and held-out evaluation |
| `src/run_mixed_experiment.py` | Controlled mixed CNN/frequency experiment |
| `src/refine_final_model.py` | Historical refinement and frequency-model selection |
| `scripts/` | Data preparation and verification tools |
| `models/checkpoints/final_frequency/` | Included trained model, calibration, history and selected-model results |
| `results/runs/` | Retained comparison metrics and duplicate controlled-frequency evidence |
| `tests/` | Software tests |
| `docs/Academic_Submission/` | Reviewed academic deliverables |

This project uses **filesystem storage**, not a database: CSV manifests/predictions, JSON records, PNG images, a PyTorch checkpoint and an ONNX detector. It has no database schema, user account system or persistent user-management service. Training videos are used to extract frames; the delivered interface analyzes still images and does not implement uploaded-video aggregation.

## Verification

From the project root, with dependencies installed:

```powershell
python -m pytest -q
python scripts/verify_app.py
```

The tests cover DCT mathematics, gradients, class-specific and zero-map Grad-CAM, crop bounds, manifest integrity/leakage controls, sampler balance and transfer initialization. They are software checks, not estimates of model accuracy. Runtime depends on the machine.

`verify_app.py` temporarily uses localhost port 7861, checks retained REAL and FAKE example uploads, rejects a blank input, checks stale-result clearing and closes its server. It writes an API verification record. The retained record states `visual_browser_check=false`; API success must not be presented as visual browser verification. Original-data verification and retraining scripts additionally require the missing source datasets/manifests.

Checkpoint integrity: SHA-256 `9ea2ddb91ebf833d658252336161935a4d3d6634823fb871a6ce214282d01aa2`. The same digest appears in both selected-model evaluation records.

## Limits and future work

This is a research prototype with known classification errors and dataset bias. Useful next steps are additional independently evaluated data, identity/generator-disjoint tests where feasible, multiple training seeds, compression/resize robustness studies and localization evaluation against annotated manipulation masks. These are proposed improvements, not delivered features. The student should understand the implementation and use the institution's required acknowledgment practices when presenting it.

## References

1. Zhou et al. (2025), *MH-FFNet: Leveraging mid-high frequency information for robust fine-grained face forgery detection*. [DOI](https://doi.org/10.1016/j.eswa.2025.127108).
2. Liu et al. (2022), *A ConvNet for the 2020s*. [Paper](https://arxiv.org/abs/2201.03545).
3. Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. [Paper](https://arxiv.org/abs/1610.02391).
4. Song et al. (2023), *Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models*. [Paper](https://arxiv.org/abs/2309.02218); [DeepFakeFace repository](https://github.com/OpenRL-Lab/DeepFakeFace).
5. RÃ¶ssler et al. (2019), *FaceForensics++: Learning to Detect Manipulated Facial Images*. [Dataset repository](https://github.com/ondyari/FaceForensics).
6. Guo et al. (2017), *On Calibration of Modern Neural Networks*. [Paper](https://arxiv.org/abs/1706.04599).
7. OpenCV Zoo, [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

