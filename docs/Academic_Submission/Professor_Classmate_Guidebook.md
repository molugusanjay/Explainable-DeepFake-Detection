# Professor and classmate guidebook

## 1. Introduce the updated project

Use the title CNN and Frequency-Based Explainable AI Framework for DeepFake Detection. Explain that this is a local research prototype. The custom network combines visual features with DCT enhancement and produces class-specific Grad-CAM. The browser also displays metadata indicators and calls a separately pretrained AI-image classifier.

A useful opening is: "We trained a CNN with frequency enhancement on a selected real/synthetic-face sample, then built an interface to inspect its predictions and attention. The current browser version adds metadata and an external visual detector. We report their roles separately because they are not one jointly trained model."

Do not begin by claiming universal Gemini or Midjourney detection. The local manifest does not identify those generators, and the external model integration has not been benchmarked as a complete system. The 98.64% score is validation accuracy; the current test result is 93.19%.

## 2. Explain the full workflow

Walk through the browser in order: receive an image path, gather metadata, obtain a CNN prediction and heatmap, run the external classifier, apply its threshold rule, and return seven outputs. The metadata message appears first but does not itself change the class.

Inside the CNN, explain one sequential backbone: normalized RGB input, first ConvNeXt stage, patch FGEM, second stage, global FGEM, remaining stages and classifier. The external model is a separate pretrained classifier that sees the original image.

Say: "The CNN explanation is computed before the override. If the rule changes the displayed label, this same heatmap still explains the CNN class, so I check heatmap_class in the details." This prevents a misleading explanation of an inconsistent display.

## 3. Explain the CNN, DCT and training changes

ConvNeXt-Tiny learns visual patterns. DCT represents each feature map using cosine coefficients. The code removes a selected low-frequency band and reconstructs a residual, then learns channel interactions between that residual and the original features. High-frequency content can also be genuine texture or compression.

Patch FGEM works on 7 x 7 blocks of the shallow 96-channel feature map. Global FGEM processes each full channel after the 192-channel stage. The final feature map is approximately 768 x 7 x 7, and the classifier produces REAL/FAKE logits.

The current FGEM output adds dropout with probability 0.1 during training. AdamW uses 4e-5 for the backbone and 4e-4 for FGEMs, with weight decay 0.02. Effective batch size is roughly 16 through accumulation of two batches of eight. The best selected epoch is six; training ended after epoch eleven.

Explain that this run starts from ImageNet. No transfer parent is recorded, so the new experiment is not evidence of continued learning from the previous mixed model.

## 4. Explain data and evaluation honestly

The collection name contains 140k, but the actual prepared subset is 2,000 images, balanced 1,000 per class. Training has 1,589 images, validation 220 and test 191. Read those counts from the retained manifest and checkpoint, not from an approximate comment in the preparation script.

The manifest content hash matches the checkpoint, and every image matched its stored hash. However, the group IDs are generated per file, not true identity labels. Identity overlap and near duplicates remain possible. The current unseeded preparation script also lacks the retained test split, so preserve the saved CSV.

Report the main result as: "The selected checkpoint reproduced 98.64% validation accuracy. On the existing 191-image test split, a separate review measured 93.19% accuracy and 93.12% balanced accuracy." The test transform directly resizes stored images; it does not benchmark the complete crop-plus-external app.

The test matrix contains 11 false alarms and two missed fakes. The 95% test target is unmet. Never present the external 99% display constant as measured accuracy.

## 5. Explain confidence, heatmaps and metadata

Despite its filename, the deployed calibrated.pt contains calibrated=False and temperature=1.0. The ordinary CNN confidence is therefore uncalibrated softmax output. Old calibration.json records belong to the previous checkpoint.

Red and blue describe CNN classes. The current overlay uses the selected rectangular crop or the whole image, with no facial oval. It is coarse attention from a roughly 7 x 7 feature map. Explain broad regions, not exact edited boundaries or specific defects.

Metadata scanning checks the first 500,000 bytes and EXIF fields. Say: "These messages are file indicators; missing EXIF or a matching word does not prove AI generation." Text matching is not a SynthID decoder or a C2PA signature validator.

The external artificial threshold is greater than 0.20. If triggered, the app forces FAKE (AI Generated) and 99% while keeping the original CNN probabilities and map. Explain the inconsistency openly and identify output separation and calibration as improvement priorities.

## 6. Prepare the demonstration

Open the project root in VS Code and select the intended Python environment. Start_App.bat automatically changes to the root and finds a local environment. Confirm the checkpoint and YuNet detector exist. External model loading may need internet on a new computer and can fail if dependencies or files are unavailable.

Use the current example originals supplied in Evidence, not an already colored heatmap. They were generated from the first current test example of each class. The figures in this documentation show the CNN path only. Live browser results can differ because of the external rule.

Rehearse a real face, a fake face, a no-face image large enough for fallback, and a very small invalid image. Do not promise blank rejection: current code analyzes blank images through the full-image path. Also rehearse the case where the CNN label and the displayed override disagree.

Open Analysis details and identify probability_real, probability_fake, confidence_type, heatmap_class and face_box. These fields help explain which output came from which stage. Never conceal a wrong result or change its label during the demonstration.

Launch the local web application from project root

```powershell
.\venv\Scripts\python.exe src/app.py --open-browser
```

Use another port if the default is occupied

```powershell
.\venv\Scripts\python.exe src/app.py --port 7861 --open-browser
```

Save CNN-only output for a current example

```powershell
.\venv\Scripts\python.exe src/predict.py docs/Academic_Submission/Evidence/current_1_original.png --output results/current_demo
```

## 7. Present the results table

Show validation and test in different columns. Validation: 98.64% accuracy, 98.51% balanced accuracy, 220 images. Test: 93.19%, 93.12%, 191 images. Test AUC is 0.9947; explain that AUC measures ranking and is not accuracy.

Explain the test confusion matrix: 83 real correct, 11 real falsely flagged fake, two fake missed and 95 fake correct. Fake recall is high at 97.94%, but fake precision is 89.62% because of false positives.

Keep historical results on a separately labelled slide. Old mixed accuracy 92.66% and the old CNN-only 93.90% came from different data. Do not subtract them from the new score as evidence that the new dataset improved the architecture.

If asked whether the project meets 95%, answer: "It exceeds 95% on selection validation, but the existing test split is 93.19%, and the complete hybrid application is not yet benchmarked."

## 8. Walk through the important code

In model.py, show Detector.forward and the two FGEM insertions. Show the DCT mask and the new dropout. In train.py, point to the two AdamW parameter groups and the strict validation-improvement check.

In prepare_ai_dataset.py, show sampling and copying, then explain why the saved manifest is the authoritative experiment artifact: the script currently only generates train/val and does not seed its randomness. Do not regenerate it in front of the audience.

In faces.py, show full-image fallback for zero/multiple detections. In explain.py, show weighted activations, zero-map handling and rectangular blending. In predict.py, show uncalibrated confidence and the returned heatmap_class.

In app.py, show the seven outputs and the 0.99 override. In ai_detector.py, show the artificial >0.20 condition. In metadata.py, show string matching. The unused ELA/FFT helpers in forensics.py are not visible features. Storage consists of files, not a database.

## 9. Discuss tests and limitations

The current review verified checkpoint identity, manifest content, image hashes, validation reproduction, test predictions, current CNN examples and blank fallback. The controlled callback check uses stubs to test the override logic, not to measure the external model.

The old verify_app.py assumes six outputs and blank rejection. Both assumptions are obsolete. Use the current verification record for the numerical test outcome, and do not present the old API pass as validation of the seven-output browser version.

Current limitations include the fixed 99% override, disagreement between label and map, no hybrid benchmark, no external-generator study, weak split provenance, uncalibrated confidence and differences between dependency requirements and the installed environment. These are concrete engineering issues that can be explained and improved.

## 10. Conclude and answer questions

A useful conclusion is: "We built and inspected a CNN-plus-frequency pipeline, trained a new small face-image experiment, and integrated metadata and a pretrained visual detector. The selected CNN has 98.64% validation accuracy and 93.19% on the existing test split. We can show its model attention while explaining the limits of the current override policy."

If asked what comes next, prioritize coherent output attribution and calibrated fusion, deterministic data preparation, current upload tests, clean-install verification and larger independent evaluations. Exact edited-pixel localization would need mask supervision and separate metrics.

Use the viva guide to understand the answers, then speak in your own words. Add the institution-required student and supervisor details. Acknowledge the base paper, datasets, external model and implementation assistance according to institutional requirements.

## Timed classroom demonstration

0:00-1:00 Introduce the system

Explain the custom CNN, Grad-CAM and the two auxiliary checks. State the study scope.

1:00-2:30 Show the architecture

Walk through the sequential CNN/DCT path and the separate external override.

2:30-4:00 Analyze a fake example

Read the actual label, confidence, metadata and CNN heatmap_class. Explain any override.

4:00-5:00 Analyze a real example

Compare the CNN probabilities with the displayed class. A false alarm remains an error.

5:00-6:00 Show fallback and invalid input

A sufficiently large no-face input is analyzed in full. A too-small input can be rejected.

6:00-7:30 Show measured results

Separate 98.64% validation from 93.19% current test accuracy. Explain 11 false alarms and two misses.

7:30-9:00 Show key source files

Show FGEM, the retained manifest and the threshold/confidence override. Distinguish code behavior from intended improvements.

9:00-10:00 Discuss limits and conclude

Prioritize accurate evidence attribution, testing and independent evaluation. Invite questions.

## Troubleshooting

### The external detector is unavailable

Its load failed or dependencies/files are unavailable.

Action: Read the message. Demonstrate the CNN as the CNN path; do not claim that external detection succeeded.

### FAKE label with a blue heatmap

The external rule can override CNN REAL while preserving the original map.

Action: Show heatmap_class and CNN probabilities. Explain the known attribution mismatch.

### Confidence is exactly 99%

The external rule assigns a constant display value.

Action: Do not call it calibrated probability or 99% accuracy. Read the external score separately.

### A blank image receives a prediction

No-face input now uses a whole-image fallback.

Action: Explain acceptance versus reliable recognition. Do not reuse the old blank-rejection demonstration.

### No colored heatmap

A zero positive Grad-CAM is possible.

Action: Keep the zero map and explain it. Never add artificial marks.

### The browser and CLI differ

The CLI runs only Predictor; the app adds auxiliary logic.

Action: Identify which path produced each result. Compare the CNN-only fields first.

### Old API verification fails

The script still expects six outputs and blank rejection.

Action: Treat it as stale test logic and plan a seven-output integration-test update.

### A new computer cannot reproduce the environment

Installed Transformers 5.19.0 differs from requirements below 5, and the old lock is historical.

Action: Use a tested compatible environment and rehearse before presenting. Do not promise that installation has been verified.

## Glossary

### CNN

A learned convolutional feature extractor.

### DCT

A cosine-transform representation of spatial variation.

### FGEM

Frequency-Guided Enhancement Module, the channel-fusion component.

### Dropout

Training regularization that randomly suppresses channels here.

### Validation

Data reused for model selection; different from final testing.

### Test split

The separately assigned 191-image subset used for this review evaluation.

### Grad-CAM

A gradient-based map of CNN class relevance.

### Calibration

Adjusting probability estimates; the current checkpoint is uncalibrated.

### Override

A rule that changes the displayed result after CNN inference.

### Metadata heuristic

A file-field/string indicator rather than authenticated provenance.

### Balanced accuracy

The mean of REAL and FAKE recall.

### Checkpoint provenance

The hashes and metadata linking weights to a specific experiment.

## Before presenting

1. Use the new checkpoint hash and current split counts.

2. Label validation, test and historical scores separately.

3. Explain the external override and its constant 99% value.

4. Identify the CNN class that each heatmap explains.

5. Rehearse real, fake, fallback and small invalid inputs.

6. Use actual observed results rather than old demo confidence values.

7. Add verified student, college, department and guide details.

8. Retain dataset/model attribution and disclose limitations.