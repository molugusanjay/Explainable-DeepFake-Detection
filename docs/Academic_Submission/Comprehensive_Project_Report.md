# CNN and Frequency-Based Explainable AI Framework for DeepFake Detection


## Abstract
This project implements an explainable image-classification prototype using ConvNeXt-Tiny and patch/global Discrete Cosine Transform enhancement. The updated system adds a sampled real/synthetic-face experiment, frequency-module dropout and differential learning rates, full-image fallback, rectangular Grad-CAM, metadata inspection and a pretrained external AI-image detector. The application combines these components through a rule rather than joint training.

The retained new manifest contains 2,000 images: 1,589 training, 220 validation and 191 test. The deployed checkpoint matches the sixth-epoch best model and is uncalibrated. Review inference reproduces 98.64% validation accuracy and measures 93.19% test accuracy, with 93.12% balanced test accuracy. The test figure applies to the custom CNN dataset transform, not the entire hybrid upload workflow.

Grad-CAM shows CNN class relevance rather than verified edited pixels. A source and controlled-callback review identifies an external override that forces a 99% display while retaining the original CNN heatmap and probabilities. The report therefore presents current capabilities, measured results, reproducibility limits and correction priorities separately. Earlier mixed-dataset scores are historical comparisons and are not attributed to the new checkpoint.


## 1. Introduction and project background

This project studies explainable classification of real and synthetic facial images using a ConvNeXt convolutional network with frequency-guided feature enhancement. A local Gradio application displays a prediction, confidence, an original-image Grad-CAM overlay and an explanation. The updated application also examines file metadata and calls a separately pretrained visual AI detector.

The 9 October 2026 revision contains a new training experiment using a 2,000-image sample from the local 140k Real and Fake Faces collection. It also changes input handling, regularization, optimization and the application decision logic. These changes extend the implemented workflow, but do not establish reliable detection of every named image generator.

Three distinct evidence layers must be kept separate: the CNN training experiment, the CNN-only evaluation pipeline, and the complete application with an external-detector override. Each has different behavior. An accuracy measured for the CNN does not automatically measure the application after its prediction has been overridden.

This report documents the files as reviewed, including observable limitations. Earlier DeepFakeFace and FaceForensics++ experiments remain historical evidence. Their old scores and calibration metadata do not describe the new deployment checkpoint.

Evidence: README.md; src/app.py; src/prepare_ai_dataset.py; current checkpoint metadata


## 2. Problem statement and objectives

The problem is to provide an inspectable research workflow for distinguishing real facial photographs from manipulated or synthetic examples. A label alone gives little information about which learned features affected a decision. The project therefore combines visual and frequency processing with class-specific attention visualization.

The implemented objectives are to load an image, choose a face crop or a full-image fallback, calculate a CNN result, display model confidence, overlay Grad-CAM and describe broad attention regions. The latest interface adds metadata messages and an external visual classifier that can override the displayed result.

The original target was at least 95% accuracy. The new saved model exceeds that number on the validation subset used for selection. A fresh review of the existing test split measured 93.19%, so the 95% test target is not established. No full-system test score is available for the override policy.

Explanations should identify model attention rather than claim exact edited pixels. The report also aims to distinguish reproducible implementation details from unsupported marketing claims, historical records and proposed future work.

Evidence: User project specification; src/app.py; revision review_predictions.csv


## 3. Existing approaches and literature

Spatial CNNs learn discriminative visual features. Frequency methods represent variation using transform coefficients. The implemented model combines these ideas inside one sequential backbone through DCT-based enhancement modules. High-frequency energy alone does not establish manipulation because genuine texture, noise and compression also affect it.

The project documentation identifies MH-FFNet as its base-paper inspiration, ConvNeXt as its CNN family and Grad-CAM as its explanation method. It remains an independent adaptation rather than the authors' official implementation. The local backbone is Tiny, and face preprocessing uses YuNet. The explicitly chosen DCT cutoff, patch size and channel-fusion equation are implementation decisions.

The newly integrated umm-maybe/AI-image-detector is a separate pretrained model. Its publisher describes a 2022 proof-of-concept for artistic AI imagery and warns about scope and newer generators. Adding it is integration of a third-party classifier, not evidence that this project's CNN was trained on Gemini or modern Midjourney outputs.

Metadata inspection is another evidence source. The local implementation performs substring and EXIF checks. It does not verify a signed provenance chain or decode an invisible watermark. A missing camera field or a matching word can occur in authentic images.


|Component|Implemented role|Evidence boundary|

|---|---|---|

|CNN + DCT|Learn facial-image labels|Dataset-dependent|

|Grad-CAM|Explain CNN class support|No pixel localization score|

|External AI detector|Whole-image visual score|Separate pretrained model|

|Metadata heuristic|Display file indicators|No authenticity proof|

Evidence: docs/MODEL_NOTES.md; src/deepfake/model.py; https://huggingface.co/umm-maybe/AI-image-detector (accessed 9 October 2026)


## 4. Changes from the previous version

The earlier deployment used the mixed DeepFakeFace/FaceForensics++ frequency checkpoint with validation-fitted temperature. The file now stored at the same deployment path is byte-identical to results/runs/frequency/best.pt from the new sample experiment. Its name still contains calibrated, but its contents record calibrated=False and temperature=1.0.

Current training starts from ImageNet weights. The checkpoint records init_checkpoint=None and transfer_parent=None. Therefore the retained evidence supports a fresh training experiment on the new sample, not continued learning from the old mixed detector or a confirmed combined training set containing all three datasets.

Code changes add Dropout2d(0.1) after each FGEM output, a tenfold learning rate for frequency-module parameters and weight decay 0.02. The predictor uses two PyTorch CPU threads. The face handler falls back to the whole image when zero or multiple faces are detected. The overlay no longer applies a facial oval.

The app adds metadata output and an external visual detector. An artificial score greater than 0.20 triggers FAKE (AI Generated) and a fixed 99% display. Existing CNN probabilities and heatmap remain unchanged. Old README paragraphs and JSON verification files still describe the previous version; they must not be used as current-system evidence.


|Item|Previous documentation|Current implementation|

|---|---|---|

|Deployment temperature|2.716439, calibrated|1.0, uncalibrated|

|No/multiple faces|Rejected|Full-image fallback|

|Overlay area|Facial oval|Full selected rectangle|

|External classifier|Absent|Pretrained detector + override|

|Training data|Mixed DFF and FF++|2,000-image new sample|

Evidence: src/app.py; src/deepfake/{model,faces,explain,predict}.py; checkpoint_audit.json


## 5. Requirements and technologies

Run from the project root with the intended Python environment. Required local assets include the deployment checkpoint and YuNet ONNX detector. The external classifier additionally needs compatible Transformers dependencies and pretrained files; first acquisition can require internet access. Prediction is implemented through local Python model calls, not a paid inference API.

The environment inspected for this review uses Python 3.14.3, CPU PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.19.0 and Gradio 6.29.1. CUDA is unavailable in this environment. This does not prove the laptop lacks a GPU; it describes the selected environment. No measured minimum memory requirement or real-time latency guarantee is supplied.

requirements.txt requests Transformers >=4.0,<5, while the inspected environment contains 5.19.0. ENVIRONMENT_LOCK.txt records the older environment. Therefore installing requirements on a new computer does not recreate the environment used in this review. Test a clean installation before a demonstration.

Core libraries are PyTorch/torchvision for training, Pillow and OpenCV for images and detection, NumPy and scikit-learn for metrics, Gradio for the interface and Transformers for the external classifier. The project uses files rather than a database. YAML files remain reference records; runtime code uses arguments and explicit Python settings.

Evidence: requirements.txt; venv/pyvenv.cfg; current_review.json; ENVIRONMENT_LOCK.txt


## 6. System architecture

The application has three analysis paths. Metadata inspection produces a text message. The custom CNN produces a class, probabilities and Grad-CAM. The external pretrained visual detector scores the original image and may change the displayed label. These branches do not jointly train or fuse features into one learned end-to-end model.

Inside the custom model, the CNN and frequency stages are sequential. An RGB input becomes a normalized 3 x 224 x 224 tensor. ConvNeXt stage 1 produces 96 x 56 x 56 features, followed by patch FGEM. Stage 2 produces 192 x 28 x 28 features, followed by global FGEM. Remaining stages end at 768 x 7 x 7 before pooling and a two-class classifier.

The class order is REAL=0 and FAKE=1. Grad-CAM differentiates the CNN's selected logit at features[7]. The external rule operates only after this CNN result has been computed. It does not regenerate the map or train a combined fusion classifier.

Metadata messages accompany the result but do not themselves set the class in the current callback. A diagram of the full application must therefore distinguish annotation, model inference and the final override rule.

[Architecture/workflow diagram appears in the PDF.]

Evidence: src/app.py; src/deepfake/model.py; src/deepfake/predict.py


## 7. Dataset inventory and provenance

The new prepared manifest contains exactly 2,000 rows: 1,000 real_photo and 1,000 ai_generated. It points to copied JPEG images under data/prepared/images. The source preparation script reads a local folder named 140k Real and Fake Faces and samples up to 1,000 files per class from its train directory. The collection name is not the number used for training.

The actual manifest has 1,589 training images, 220 validation images and 191 test images. Its SHA-256 matches the value stored in the new checkpoints. Every referenced image exists and matched its recorded content hash during review. The repository's manifest loader accepted the current rows and split/class constraints.

The current preparation script uses random.shuffle and random.random without setting a seed. It only assigns train and val, whereas the retained manifest also contains test rows. Thus that script alone cannot reproduce the exact saved split. Training seed 42 does not retroactively seed the separate data preparation process.

Group and source identifiers are generated from each class and enumeration index. They are not verified identities or original source-image relationships. Exact-hash integrity checks are useful but cannot establish identity-disjointness or eliminate near duplicates. Preserve the actual manifest instead of regenerating it after examining scores.


|Split|REAL|FAKE|Total|

|---|---|---|---|

|Training|805|784|1,589|

|Validation|101|119|220|

|Test|94|97|191|

|All selected images|1,000|1,000|2,000|

Evidence: src/prepare_ai_dataset.py; data/prepared/manifest.csv; checkpoint_audit.json


## 8. Preprocessing and input handling

Dataset evaluation uses the saved prepared images directly: resize to 224 x 224, convert to a tensor and normalize with ImageNet RGB means [0.485,0.456,0.406] and standard deviations [0.229,0.224,0.225]. The training transform additionally uses random horizontal flips, occasional Gaussian blur and occasional small Gaussian noise.

Application inference first corrects orientation and converts to RGB. YuNet runs at score threshold 0.85, NMS threshold 0.3 and topK 5000. Detection images are limited to a maximum side of 960 pixels. A single sufficiently large face receives a 10% margin, clipped to image bounds, before resizing.

If zero or multiple faces are detected, current code returns the whole image and its full bounds. It still rejects an image whose shorter side is below 48 pixels, or a single detected face below 48 pixels. A blank 256 x 256 image was actually analyzed through the fallback during review. The old blank-rejection claim is obsolete.

Training preparation copies selected files without invoking YuNet. Direct dataset evaluation consequently differs from the application's optional recropping. The reported new CNN test score measures the dataset transform, not the entire upload pipeline. Whole-image acceptance also does not demonstrate reliable non-face classification.

Evidence: src/deepfake/data.py; src/deepfake/faces.py; src/prepare_ai_dataset.py; current_review.json


## 9. CNN implementation and tensor flow

Detector supports Tiny and Base backbones, but the packaged checkpoint uses Tiny with both frequency modules. Its configuration is backbone=tiny, mode=both, patch_size=7 and cutoff=0.25. The instantiated selected model has 28,007,138 parameters, of which 185,472 belong to the two FGEM modules.

The network uses torchvision ConvNeXt feature blocks, global average pooling and a classifier whose final linear layer has two outputs. CNN-only mode replaces both frequency modules with identity operations. The existence of patch-only and global-only options does not mean complete measured ablations for those modes are included.

During normal training, the backbone starts from ImageNet weights. During inference, the architecture is constructed without requesting pretrained weights and then receives the saved trained parameters. load_detector checks the class order, positive training-step count and architecture-compatible state dictionary. It uses restricted weights_only checkpoint loading.

The predicted class follows the largest logit. A positive scalar temperature preserves that class ordering. Therefore confidence calibration changes probability values while leaving classification unchanged, except for irrelevant implementation-level numerical tie considerations.


|Position|Tiny feature shape|Next operation|

|---|---|---|

|Input|3 x 224 x 224|Stem and first stage|

|Shallow stage|96 x 56 x 56|7 x 7 patch FGEM|

|Intermediate stage|192 x 28 x 28|Global FGEM|

|Remaining stages|384 x 14 x 14; 768 x 7 x 7|Average pooling|

|Classifier|2 logits|REAL / FAKE probabilities|

Evidence: src/deepfake/model.py; src/deepfake/checkpoints.py; checkpoint audit


## 10. DCT frequency processing

An orthonormal DCT-II represents each spatial feature array using cosine basis functions. The code caches CPU basis matrices and moves them to the tensor device. For an H x W feature array X, it computes coefficients C = D_H X D_W^T.

The mask retains coefficient (u,v) when the average of u/max(H-1,1) and v/max(W-1,1) is greater than 0.25. This triangular normalized-frequency rule is not equivalent to removing exactly 25% of coefficients. Inverse reconstruction is R = D_H^T (C elementwise-multiplied by mask) D_W.

Patch FGEM reshapes the shallow map into nonoverlapping 7 x 7 feature patches, filters each patch and restores the original layout. Global FGEM filters the entire spatial extent of each intermediate feature channel. Neither operation directly labels pixels as authentic or manipulated.

DCT and inverse DCT run in float32 even inside mixed-precision training, reducing numerical problems. The operation remains differentiable. Software tests compare it with an independent SciPy DCT implementation and check that a constant low-frequency signal is suppressed.

Evidence: src/deepfake/model.py: dct_basis, frequency_residual, FGEM.frequency; tests/test_core.py


## 11. Frequency-guided enhancement and regularization

Each FGEM produces a frequency residual R from spatial features X. Learned 1 x 1 convolutions create Q from X and K,V from R. Flattened spatial positions form the feature dimension. Channel attention is softmax(Q K^T / sqrt(C H W)), and attention @ V is reshaped into the original feature-map dimensions.

The module adds that fused result to X, applies an output 1 x 1 convolution, then applies Dropout2d with probability 0.1. This is a C x C channel-attention matrix, not a pixel-to-pixel self-attention map. Patch and global modules use the same fusion structure at different backbone depths.

Dropout is active in training and disabled in model.eval(). It adds no learned weights, so the model still has 28,007,138 parameters. Initializing the final projection as identity does not make the complete FGEM identity: the frequency fusion is still added.

DCT basis construction, coefficient masking and inverse reconstruction are fixed differentiable operations. CNN and convolutional projection weights are learned. Float32 transform/attention calculations support numerical stability when mixed precision is enabled on compatible CUDA runs.

[Architecture/workflow diagram appears in the PDF.]

Evidence: src/deepfake/model.py; current checkpoint configuration


## 12. New training experiment

The saved run records an ImageNet-initialized ConvNeXt-Tiny with both frequency modules, batch size 8, accumulation 2, two data-loader workers, seed 42, nominal learning rate 4e-5, a maximum of 20 epochs and patience 5. The checkpoint has no transfer parent. It therefore does not demonstrate continued training from the earlier mixed model.

Current code uses AdamW with betas (0.9,0.999) and weight decay 0.02. The backbone receives the nominal rate and the patch/global FGEM parameters receive ten times that rate, 4e-4. Cross-entropy is the objective; gradients are clipped to norm 1.0. Accumulation approximates a batch of 16 while allowing a smaller in-memory batch.

The manifest has no dataset column, so sampling/selection use the available class/method information rather than a macro average over multiple named datasets. The validation selection score is balanced accuracy. Eleven epochs are retained. Epoch 6 first reached the best score, and the equal score at epoch 10 did not replace it because improvement must be strict. Five non-improving epochs ended the run at epoch 11.

The deployed best checkpoint records 600 optimizer steps. The last checkpoint records 1,100. The historical old run stored beside it is not the new training history. Use results/runs/frequency/history.json for the new experiment.

Evidence: src/train.py; results/runs/frequency/history.json; best.pt/last.pt metadata


## 13. Checkpoint identity and confidence

The deployment file models/checkpoints/final_frequency/calibrated.pt is byte-identical to the new results/runs/frequency/best.pt. Both have SHA-256 317e1c1e9a73eaeff6d1454ec5e926775597ad23e612e390eda99d4d3f7f1140. Its internal epoch index is 5, meaning the sixth human-counted epoch.

Its metadata says temperature=1.0 and calibrated=False. CNN probabilities are softmax(logits/1.0), so they are uncalibrated softmax scores. The old calibration.json, old temperature 2.716439 and old calibrated-confidence descriptions refer to a different checkpoint. A filename does not establish calibration.

Predictor displays a low-confidence message below 70%. This is a UI heuristic, not a validated rejection policy. It returns a class for any input accepted by the current cropper, even when uncertain.

The external override changes result confidence to 0.99 rather than using either model's calibrated probability. That constant is not supported as a 99% correctness rate. probability_real, probability_fake, heatmap_class and confidence_type remain those of the CNN, creating potentially inconsistent result metadata.


|Field|Current value|Interpretation|

|---|---|---|

|Checkpoint|Same bytes as new best.pt|Epoch 6 selected|

|Temperature|1.0|No temperature adjustment|

|calibrated|False|Uncalibrated CNN output|

|Override confidence|0.99 constant|Rule-assigned display|

Evidence: checkpoint_audit.json; src/deepfake/predict.py; src/app.py


## 14. Grad-CAM and current overlay

Grad-CAM registers a forward hook at features[7], differentiates the selected CNN logit with respect to those activations, averages gradients spatially to obtain channel weights, sums weighted activations and retains positive support with ReLU. The hook is removed afterwards. A map whose maximum is effectively zero remains zero.

The native map is about 7 x 7. It is resized to the selected rectangle. Current face_overlay no longer multiplies by a facial oval. The rectangle can be a detected-face crop or the complete image under fallback. Pixels outside a face rectangle remain unchanged. Within it, heat at least 0.35 is blended with alpha=0.60 x heat.

Red displays positive CNN FAKE support and blue positive CNN REAL support. The text divides the map into a 3 x 3 grid, ranks mean attention and names up to two regions. It is template logic, not an LLM or a learned detector of texture defects.

Some strings still say facial area or approximate facial oval even when current code uses an entire rectangular image. More importantly, an external FAKE override can retain a blue REAL CNN map. Always state which model and class a heatmap explains. No pixel-level manipulation-mask accuracy was evaluated.

Evidence: src/deepfake/explain.py; src/deepfake/predict.py; controlled callback check


## 15. Metadata background check

analyze_metadata scans the first 500,000 bytes when given a file path. It performs case-insensitive substring matches for names such as Midjourney, DALL-E, Stable Diffusion, SynthID, c2pa, Google and Gemini. It also uses Pillow to inspect EXIF Make, Model, LensModel and Software fields and reports missing camera metadata.

These are lightweight heuristics. The presence of a generic word such as Google does not establish generated content. Conversely, an AI image can have stripped or altered metadata. Authentic web images may lack EXIF. The displayed warnings are indicators for inspection, not a validated independent classification result.

Searching for the text SynthID does not implement invisible watermark detection. Searching for c2pa does not authenticate a provenance manifest or check its signature. The report describes the actual byte/string checks and does not infer those additional capabilities.

The callback displays metadata first, but only the external visual-detector message triggers its FAKE override. Metadata findings alone do not change the CNN class in the inspected implementation.

Evidence: src/deepfake/metadata.py; src/app.py


## 16. External visual detector and decision rule

get_ai_classifier creates a Transformers image-classification pipeline using umm-maybe/AI-image-detector. The app attempts to preload it at import/startup and caches the returned object. A load failure stores FAILED; subsequent calls report unavailability. Initial loading may require downloaded files and compatible dependencies.

detect_ai_generation searches the returned labels for artificial. A score strictly greater than 0.20 produces a message containing STRONG VISUAL AI SIGNATURE DETECTED. app.py detects that phrase, assigns FAKE (AI Generated), sets confidence to 99%, and prepends the external message to the CNN explanation. A score equal to 0.20 does not trigger this rule.

This is a hand-coded decision policy, not learned probability fusion or validation-calibrated ensembling. The 20% threshold and 99% constant are present in code, but no threshold study or complete-system held-out evaluation accompanies them. A low trigger can increase sensitivity while increasing false positives.

The publisher describes this model as a proof-of-concept for artistic images, trained in 2022, with limitations on newer generators and photographic use. Its published validation numbers belong to that external model and dataset. They are not local project scores and are not included in the project results table.

Evidence: src/deepfake/ai_detector.py; src/app.py; https://huggingface.co/umm-maybe/AI-image-detector (accessed 9 October 2026)


## 17. Software modules and storage

src/app.py builds the Gradio interface and combines results. src/deepfake/predict.py loads the custom model, crops an image, computes Grad-CAM and produces the CNN result dictionary. metadata.py and ai_detector.py add the new auxiliary checks. faces.py and explain.py control fallback and visualization.

The new forensics.py contains JPEG error-level-analysis and FFT-spectrum helpers, but the current app and predictor do not call them. Their existence in a source file does not make ELA or a displayed FFT spectrum an implemented application output. The model's actual frequency mechanism remains intermediate-feature DCT.

There is no application database. CSV files hold manifests and retained predictions, JSON files hold histories/results, .pt files hold model weights/metadata and PNG files hold images. Gradio uses temporary upload/output files. No persistent user-account or analysis-history service is implemented.

CLI prediction through src/predict.py uses Predictor alone. It saves an original image, CNN overlay and prediction JSON, but does not execute app.py's metadata or external override. CLI and browser results may consequently differ for the same input.


|Module|Purpose|

|---|---|

|src/deepfake/model.py|CNN, DCT, FGEM, dropout|

|src/deepfake/predict.py|CNN inference and heatmap|

|src/deepfake/metadata.py|Byte and EXIF heuristics|

|src/deepfake/ai_detector.py|External visual model|

|src/app.py|Seven outputs and override|

|src/prepare_ai_dataset.py|Sample/copy manifest helper|

Evidence: src/app.py; src/predict.py; src/deepfake/*.py


## 18. Application workflow and demonstration

Start from the project root using Start_App.bat or the intended environment's python src/app.py --open-browser. The default server is localhost port 7860 with share=False. The external model preload can affect startup time; the custom Predictor is still initialized lazily on first analysis.

The callback returns seven outputs: metadata text, displayed class, confidence text, original image, CNN overlay, explanation and a JSON summary. The external classifier runs after CNN inference. It analyzes the whole input while the CNN may have processed a cropped face.

During a demonstration, show metadata as context, then name the CNN result and inspect the heatmap class. If the external rule overrides the result, explain the rule and its fixed confidence. Do not describe an unchanged CNN heatmap as an explanation of the external model.

Missing or small inputs can trigger handled errors and clear outputs. A blank image of sufficient size now follows the full-image fallback rather than being rejected. The old six-output verify_app.py and its blank rejection expectation are outdated. The guidebook provides a current seven-output demonstration checklist.

[Architecture/workflow diagram appears in the PDF.]

Evidence: Start_App.bat; src/app.py; scripts/verify_app.py; current_review.json


## 19. Validation results and model selection

The new run's best selected checkpoint reaches 98.6364% validation accuracy, 98.5149% balanced accuracy and ROC AUC 0.9997504 on 220 images. The confusion matrix is [[98,3],[0,119]], with rows REAL/FAKE and columns predicted REAL/FAKE. There are three false positives and no false negatives at the default threshold.

The README headline 98.18% appears at other epochs, including epoch 11, but the deployed best checkpoint is epoch 6. Fresh CPU inference using the same saved validation images reproduced its accuracy, balanced accuracy, AUC and confusion matrix. That confirms the saved model/record relationship for this transform.

Validation selected the checkpoint, so this score is not an untouched test result. Eleven training epochs and repeated selection on the same 220 images can favor that subset. The balanced score matters because validation includes 101 real and 119 fake examples.

This is the custom CNN's validation performance. It does not measure the external detector, metadata heuristic, face-detection fallback, crop preprocessing or the combined application's override behavior.


|Validation measure|Selected epoch 6|

|---|---|

|Accuracy|98.64%|

|Balanced accuracy|98.51%|

|ROC AUC|0.9998|

|FAKE precision|97.54%|

|FAKE recall|100.00%|

|Images|220|

Evidence: results/runs/frequency/history.json; checkpoint_audit.json; current_review.json


## 20. Independent review of the saved test split

On 9 October 2026, the review evaluated the already selected checkpoint on the manifest's existing test split, without training, calibration or threshold tuning. Inputs used the same direct resize/normalize transform as dataset evaluation. The manifest and checkpoint hashes matched, and all source image hashes were valid.

The 191-image test contains 94 real and 97 fake examples. Accuracy is 93.1937%, balanced accuracy 93.1180%, ROC AUC 0.9947357, fake precision 89.6226%, fake recall 97.9381% and fake F1 0.93596. The observed 95% accuracy target is not reached on this split.

This is a newly computed review result, separately saved with its per-image predictions. It is not a score that was found in the old historical evaluation directory. No identity-disjointness, unseen-generator coverage or previously untouched selection history is assumed for the small test set.

The review did not evaluate every upload through YuNet and the external rule. Therefore this is a CNN dataset-transform result, not full-application accuracy. The validation-to-test gap also prevents describing the project simply as a 98.64%-accurate detector.


|Measure|Validation|Test|

|---|---|---|

|Accuracy|98.64%|93.19%|

|Balanced accuracy|98.51%|93.12%|

|ROC AUC|0.9998|0.9947|

|FAKE precision|97.54%|89.62%|

|FAKE recall|100.00%|97.94%|

|Images|220|191|

Evidence: Review evidence/current_review.json and review_predictions.csv; checkpoint and manifest SHA-256


## 21. Confusion matrix and error interpretation

Rows show the supplied dataset label and columns show the CNN prediction. The current test matrix is [[83,11],[2,95]]. The model correctly labels 83 real and 95 fake images. Eleven real images are flagged fake and two fake images are missed, giving 178 correct out of 191.

Real recall is 83/94 = 88.30%; fake recall is 95/97 = 97.94%. Balanced accuracy is the mean of those two recalls, 93.12%. A false accusation and a missed fake are different failure types, so both should be discussed in a viva.

AUC evaluates ranking across thresholds and can remain high even when the fixed operating point produces errors. AUC 0.9947 must not be called 99.47% accuracy. Altering the threshold after inspecting this test would require a separately defined validation procedure and a new independent evaluation.

The confusion matrix uses publisher/class-folder labels as supplied. It does not independently authenticate those labels, and the small sample cannot establish broad real-world reliability.

Evidence: Review evidence/review_predictions.csv; current_review.json


## 22. Current CNN heatmap examples

The paired images below were regenerated during this review with the new deployment checkpoint and current Predictor code. The examples are the first manifest test image in each class, rather than hand-selected successes from a large search. Their purpose is to show the actual prediction/overlay workflow.

The illustrated fake sample receives a CNN FAKE prediction with approximately 99.57% uncalibrated confidence. The overlay is red and extends across the selected rectangular crop according to the positive activation map. The paired original and overlay preserve the photograph's context.

These are CNN-path examples, not screenshots of the complete hybrid web application. The external detector and metadata rule were not applied to these example outputs. A current real-class example is also included in the presentation and guidebook.

A Grad-CAM map is low-resolution class relevance. It cannot certify that every highlighted pixel was altered, name a generator, or prove a specific visual defect. Confidence and the apparent clarity of a map do not establish correctness.

Evidence: current_review.json examples; regenerated current_1_original.png and current_1_heatmap.png


## 23. Testing and regression findings

The current review checks checkpoint identity, manifest hashes, referenced images, validation reproduction, existing-test predictions and current CNN examples. It also exercises the saved app callback with controlled stubs to isolate the external override rule. That controlled check is a software diagnostic, not an external-model accuracy benchmark.

When the stub supplies CNN REAL at 80% and an external artificial message at 21%, the exact callback returns FAKE (AI Generated) at 99% while keeping the blue CNN overlay, REAL heatmap_class and original 80%/20% CNN probabilities. This confirms the inconsistency predicted by source inspection.

The legacy upload verifier expects six outputs, but the app now returns seven. It also expects blank-image rejection, while current cropper code uses the whole blank image. The previously saved passing API record therefore cannot validate this version. Updated end-to-end tests are required before asserting full application correctness.

The software suite covers DCT, gradients, Grad-CAM, data integrity, sampling, transfer and local workflow routing. All 20 tests passed in 54.78 seconds in the reviewed environment. REVIEW_AND_VERIFICATION.md records the execution scope. Passing numerical tests does not establish reliable AI-image detection or validate the newly added external policy.

Evidence: tests/; scripts/verify_app.py; current_review.json controlled_override_check; pytest logs


## 24. Historical results and fair comparisons

The retained historical CNN-plus-frequency model achieved 92.66% combined legacy accuracy and 89.24% additional DFF accuracy. The earlier CNN-only baseline achieved 93.90% combined legacy accuracy. Those records identify the old checkpoint hash beginning 9ea2ddb9, not the current checkpoint beginning 317e1c1e.

The new sample test and the historical mixed test differ in images, class distributions, split construction, initialization/training details and preprocessing. Their percentages must not be interpreted as a controlled improvement of one architecture over another.

The current checkpoint has no transfer parent and its manifest contains only real_photo/ai_generated rows without DFF/FFPP dataset tags. The evidence therefore does not show that old and new data were jointly used in this latest training run.

Old calibration.json, selection.json, history.json and demo records remain beside the new deployment filename. Keep their provenance explicit or archive them separately before future releases. A reproducible package should bind every result to its actual checkpoint, manifest and code version.


|Experiment|Accuracy|Scope|

|---|---|---|

|Old mixed frequency|92.66%|2,901 legacy test images|

|Old additional DFF|89.24%|1,320 DFF images|

|New selected CNN|98.64%|220 validation images|

|New review CNN test|93.19%|191 current test images|

Evidence: Historical test_metrics.json; checkpoint_audit.json; current_review.json


## 25. Challenges and limitations

The project combines useful components, but the current hybrid result has an unresolved explanation mismatch and a hard-coded confidence value. Metadata strings can cause false alarms. The external threshold lacks local validation, and the publisher's detector scope does not establish reliability on modern photographic generators.

The new sample is small, source/identity provenance is weak, preparation is not exactly reproducible from the current script, and the observed test score is below 95%. Broad dataset acceptance under full-image fallback is not equivalent to trained non-face capability. Training images and app crops also follow different paths.

The deployment checkpoint is uncalibrated. Historical sidecar records are stale. The current 20-test software pass does not cover the full hybrid policy. The older six-output API record is obsolete after the behavior changes. The local dependency range differs from the installed Transformers version, creating a portability risk.

No full-system held-out benchmark, latency study, localization-mask evaluation, robustness/fairness audit, identity-disjoint split or multi-seed study is available. State these boundaries directly while showing what the implemented pipeline does.

Evidence: Source review; checkpoint/manifest audit; current test and controlled callback check


## 26. Future improvements

First separate the output into named evidence sources: CNN label/probabilities/Grad-CAM, external detector label/score and metadata indicators. If an override remains, preserve the originating model and label rather than claiming its fixed display confidence is calibrated. Either explain the final decision with the corresponding model or explicitly label the existing map as CNN-only.

Fit any fusion rule or threshold on validation data and evaluate it once on a separately reserved test. Recalibrate only after choosing the model, retain checkpoint hashes, and avoid overwriting historical metadata. Record the actual data seed, source provenance and preprocessing policy so the saved split can be recreated.

Use larger approved data, multiple seeds, external-generator testing and checks for identities/near duplicates. Compare the new architecture and CNN-only baseline under the same protocol. Evaluate face handling and full-image fallback separately, including rejected/ambiguous inputs.

Update the seven-output upload tests and clean-install environment lock. If exact manipulation localization is required, train a supervised localization component with annotated masks and localization metrics. Video aggregation, persistent history and verified provenance are separate future features.

Evidence: Proposals derived from current implementation findings; not implemented features


## 27. Conclusion

The updated project implements a local CNN-plus-frequency research detector with current face/full-image input handling, Grad-CAM, metadata heuristics and a separately pretrained visual-detector integration. The new sample experiment and retained model records can be inspected and reproduced at inference level.

The deployed checkpoint reproduces 98.64% accuracy on its 220-image validation subset. A fresh check of the existing 191-image test split gives 93.19%, so a 95% test-accuracy claim is unsupported. The complete hybrid application has not received a separate accuracy evaluation.

The strongest presentation is a precise account of the implementation and its evidence: distinguish the CNN from the external rule, validation from test, model confidence from correctness and heatmap relevance from edited pixels. Known inconsistencies should be acknowledged and repaired in a future implementation revision rather than hidden in the documentation.

These academic materials describe the current files without retraining or modifying model behavior. Add verified student, roll-number, college and supervisor details using the institution's format before formal submission.

Evidence: Current reviewed source and experiment evidence, 9 October 2026


## 28. References and evidence map

[1] Kai Zhou, Guanglu Sun, Jun Wang, Linsen Yu and Tianlin Li. MH-FFNet: Leveraging mid-high frequency information for robust fine-grained face forgery detection. Expert Systems with Applications 276, 127108, 2025. https://doi.org/10.1016/j.eswa.2025.127108

[2] Haixu Song, Shiyu Huang, Yinpeng Dong and Wei-Wei Tu. Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models. 2023. https://arxiv.org/abs/2309.02218. Dataset and pairing metadata: https://github.com/OpenRL-Lab/DeepFakeFace

[3] Andreas Rossler and colleagues. FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV 2019. https://github.com/ondyari/FaceForensics

[4] Ramprasaath R. Selvaraju and colleagues. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017. https://arxiv.org/abs/1610.02391

[5] Zhuang Liu and colleagues. A ConvNet for the 2020s. CVPR 2022. https://arxiv.org/abs/2201.03545

[6] Chuan Guo, Geoff Pleiss, Yu Sun and Kilian Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599

[7] OpenCV Zoo. YuNet face detection model. https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

[8] Local implementation evidence: src/deepfake/*.py, src/app.py, src/train.py, data/prepared/manifest.csv, results/runs/frequency/history.json and the current checkpoint metadata. Older calibration/selection files and prediction CSVs provide historical evidence only. Bibliographic entries above follow the sources recorded in the project.

[9] umm-maybe. AI-image-detector model card. https://huggingface.co/umm-maybe/AI-image-detector. Accessed 9 October 2026. This source describes the third-party detector, not locally measured project performance.

[10] Local collection named 140k Real and Fake Faces, with source preparation path in src/prepare_ai_dataset.py. Publisher page: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces. The exact downloaded archive provenance is not recorded in the supplied manifest; the review does not assume it contains Gemini or Midjourney examples.

[11] Review evidence bundle: checkpoint_audit.json, current_review.json and review_predictions.csv. These record the inspected model, manifest and new CNN evaluation. Source-file hashes and review boundaries accompany the submission materials.

Evidence: Existing bibliography; primary external model card; local source/evidence records


## 29. Reproduction and demonstration checklist

From the project root, use the intended local environment to run python src/app.py --open-browser. Check checkpoint and detector files first. The app attempts to load the external model; internet may be needed if its files are not already cached. Use --port 7861 if port 7860 is occupied.

For a CNN-only output, run python src/predict.py with a photograph path and --output pointing to a new directory. This does not execute the browser callback's metadata/external override. Use the current demo examples supplied in the evidence bundle to rehearse, reading the actual returned scores rather than memorizing old outputs.

Preserve data/prepared/manifest.csv and its images. Do not run prepare_ai_dataset.py casually: it writes to the same path and cannot reconstruct the retained three-way split as currently written. Training is not required for demonstration. Keep old and new model records separate.

For review results, inspect the evidence JSON and per-image predictions. The test score was produced by the saved selected checkpoint with no subsequent tuning. Student/college details remain to be supplied. Rehearse the actual browser and slide show on the presentation machine, and follow institutional rules for acknowledging assistance.

Evidence: Start_App.bat; src/app.py; src/predict.py; src/prepare_ai_dataset.py; review evidence