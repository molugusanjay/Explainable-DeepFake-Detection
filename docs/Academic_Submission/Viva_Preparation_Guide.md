# Viva preparation guide

60 questions grounded in the reviewed project.

## Purpose and scope

### 1. What does the updated project do?

It runs a ConvNeXt-Tiny plus DCT detector, shows CNN confidence and Grad-CAM, and adds metadata messages and a separately pretrained AI-image detector. The web app can override the CNN label using an external-score rule. That complete hybrid policy has no measured accuracy result yet.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 2. What problem does it address?

A binary detector alone gives a label without showing which regions affected that decision. This project combines visual CNN features with frequency enhancement and a gradient-based explanation so a user can inspect the model's output. It does not prove image authenticity or provide a forensic certificate.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 3. Does it reliably detect every generator or arbitrary image?

No. The input handler now accepts a whole image when no single clear face is detected, but accepting an input does not prove generalization. The current local experiment uses a small face-image sample. No generator-specific Gemini, Midjourney or DALL-E test is supplied, and uploaded-video analysis is absent.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 4. Is it an exact reproduction of the base paper?

No. This is an independent implementation inspired by MH-FFNet. It uses ConvNeXt-Tiny, patch/global DCT enhancement, channel fusion, YuNet and a local interface. Temperature-calibration code exists, but the current deployed checkpoint is uncalibrated. Paper results are not this project's results.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 5. What changed since the earlier project version?

A new 2,000-image sample experiment replaced the deployment checkpoint. FGEM now has dropout, training uses separate backbone/frequency learning rates, no/multiple-face inputs fall back to the full image, and the oval display mask was removed. The web app adds metadata and an external visual-detector override.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Architecture and preprocessing

### 6. What happens from upload to result?

The app gathers metadata, then the CNN corrects RGB orientation, crops one face or uses the full image, resizes to 224 x 224 and computes logits plus Grad-CAM. The external visual detector scores the original image. Above its 0.20 artificial threshold, the callback changes the label to FAKE (AI Generated) and display confidence to 99% without regenerating the CNN heatmap.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 7. Why use a CNN, and which CNN is implemented?

The detector uses torchvision's ConvNeXt-Tiny as the deployed backbone. CNN layers learn visual feature maps that are used for classification and Grad-CAM. ConvNeXt-Base is selectable in the model constructor, but the saved final model configuration is Tiny. The choice is documented as an adaptation for limited local GPU resources; no separate speed comparison is reported.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 8. Is the system one model or several models?

The custom CNN and frequency modules form one sequential network. The application additionally invokes a separate pretrained visual classifier. Its score is combined by a hand-coded override rule, not by joint feature fusion or end-to-end training. Metadata is a separate heuristic message.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 9. What are the feature dimensions and class labels?

For a 224×224 input, the Tiny backbone's first stage produces 96 channels at 56×56 and the second stage produces 192 channels at 28×28. Patch FGEM acts at the first location and global FGEM at the second. The final stage is approximately 768×7×7 before pooling. The classifier outputs two logits in the fixed order REAL=0 and FAKE=1.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 10. Why detect a face, and when is the whole image used?

YuNet selects one face with score threshold 0.85 and NMS 0.3, then adds a 10% crop margin. When it finds zero or multiple faces, the current code uses the entire image. These are alternative preprocessing paths, not evidence that the classifier was trained for every image category.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 11. Which inputs are actually rejected now?

Missing input, an image shorter than 48 pixels on one side, or a single detected face smaller than 48 pixels can trigger an error. No face or multiple faces now cause full-image analysis. A blank 256 x 256 image was analyzed in review, so the old claim that all blank images are rejected is no longer correct.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 12. How is the face image normalized?

The cropped RGB image is resized to 224×224, converted to a tensor and normalized using means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. Training additionally uses horizontal flips, occasional Gaussian blur and occasional small noise. Validation and inference do not use these random augmentations.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 13. What is the difference between face detection and fake detection?

YuNet finds facial bounding boxes; it does not determine image authenticity. The CNN-frequency classifier computes REAL/FAKE logits and uncalibrated probabilities. The app can later override the displayed result with its external-model rule. YuNet's face-detection score is separate from both classification outputs.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Frequency processing and attention

### 14. What is the frequency domain in this implementation?

It is a representation of learned feature maps using 2D cosine basis functions. Slowly varying components occupy lower DCT indices and faster changes occupy higher indices. The code filters feature-map coefficients, then reconstructs a residual. It does not classify a standalone FFT spectrum of the original photograph.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 15. Why use DCT?

The implementation uses an orthonormal DCT-II and its inverse to represent spatial variation with real-valued coefficients and remove a chosen low-frequency band. This is the selected design, consistent with the paper-inspired method. The repository does not include an FFT-versus-DCT experiment, so it does not establish that DCT is empirically better here.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 16. What does the cutoff 0.25 actually mean?

For a feature map with height H and width W, a coefficient is retained when (u/(H−1)+v/(W−1))/2 > 0.25. The other coefficients are zeroed before inverse DCT. This creates a triangular low-frequency removal mask. It does not mean exactly 25% of all coefficients are removed.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 17. How do patch and global frequency modules differ?

Patch FGEM divides each shallow feature channel into nonoverlapping 7×7 patches and filters each independently. Global FGEM applies the same transform and cutoff over each complete intermediate feature channel. The former uses local feature regions; the latter uses the full spatial extent at its stage. Both outputs are fused with visual features.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 18. Is DCT trainable, and can gradients pass through it?

The cosine basis and cutoff mask are fixed mathematical operations rather than learned parameters. Matrix multiplication, masking and inverse transformation are differentiable with respect to the input features. Gradients can therefore pass through to the CNN and the learned 1×1 projection layers. Tests compare the transform with SciPy and verify finite nonzero gradients in both modules.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 19. How does FGEM attention and dropout work?

Queries come from spatial features X, keys and values from the DCT residual R. The module computes channel attention softmax(QK^T/sqrt(CHW)), fuses V, adds X, applies a learned 1 x 1 projection and Dropout2d(0.1). Dropout is disabled during evaluation. Attention is C x C, not spatial-pixel attention.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 20. Do high frequencies or the extra detector prove an image is fake?

No. Genuine texture and compression also contain high frequencies. The CNN learns a dataset-dependent association. The external classifier has a separate scope and an unvalidated local threshold. Neither source by itself proves authenticity or a particular generator.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Grad-CAM and explanation

### 21. How is Grad-CAM calculated?

A forward hook captures the final feature stage. The code differentiates the predicted-class logit with respect to its activations, averages each channel's gradients spatially, forms a weighted sum of activation channels and applies ReLU. A positive map is divided by its maximum; a zero or nonpositive map stays zero. The hook is removed afterwards.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 22. What do red and blue mean after the new override?

They still describe the CNN map: red is CNN FAKE support and blue is CNN REAL support. If the external rule changes the displayed label, the existing map does not change. A FAKE display can therefore accompany a blue REAL CNN map. Explain the originating model and class rather than claiming both outputs agree.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 23. Does the heatmap locate the exact manipulated pixels?

No. The final feature map is coarse, approximately 7×7 before interpolation. Grad-CAM indicates regions associated with a class score, not a learned segmentation mask. The project does not evaluate pixel-level overlap against ground-truth manipulation masks and cannot claim localization accuracy.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 24. Is the heatmap still restricted to a facial oval?

No. The oval masking code was removed. Heat is blended in the whole selected rectangle, which is a face crop or the full image. Alpha is 0.60 times heat for heat at least 0.35. Some returned text still says oval or facial area, so it is stale metadata rather than the current mask implementation.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 25. How is the written explanation generated?

The displayed heatmap is divided into a 3×3 grid. The code ranks grid cells by mean heat and names up to two positive regions, such as middle central or upper left. It appends a fixed description of the classifier and an interpretation caution. There is no language model, artifact-type classifier or separate diagnosis of eyes, skin pores or blending defects.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 26. Can a wrong prediction have a convincing heatmap?

Yes. Grad-CAM explains the CNN score even when the CNN is wrong. There is also an additional mismatch if the external detector changes the displayed label while retaining a map for the CNN class. No heatmap is proof of an edit.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Training, datasets and model selection

### 27. Which datasets belong to the current and historical experiments?

The previous experiments used DeepFakeFace and FaceForensics++ subsets. The current checkpoint records a fresh ImageNet-initialized run on a 2,000-image sample from the local 140k Real and Fake Faces collection. It has no transfer parent, and the current manifest does not contain DFF/FFPP dataset tags.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 28. Was the model trained on 140,000 images?

No. The collection is named 140k Real and Fake Faces, but the retained prepared sample has 2,000 images: 1,000 real and 1,000 fake. Training uses 1,589, validation 220 and test 191. The class counts are train 805/784, validation 101/119 and test 94/97.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 29. How strong are the current data-separation guarantees?

The manifest loader checks file hashes, paths and identifiers across splits. All 2,000 referenced images matched their hashes in review. However, generated per-file source/group IDs are not true identity provenance, and the unseeded preparation script only makes train/val while the retained manifest includes test. Identity and near-duplicate separation are not established.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 30. How are training images sampled in this run?

The weighted sampler balances available class/method groups. This manifest has no dataset column, so the current run does not perform a multi-dataset macro comparison like the old mixed experiment. Sampling is with replacement; test counts stay unchanged.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 31. What are the updated training settings?

The saved run uses ConvNeXt-Tiny with both FGEMs, ImageNet initialization, seed 42, batch 8, accumulation 2, workers 2, a 20-epoch maximum and patience 5. Current AdamW code uses 4e-5 for the backbone, 4e-4 for FGEMs and weight decay 0.02. FGEM dropout is 0.1. Eleven epochs are retained.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 32. Why use accumulation, mixed precision and gradient clipping?

Accumulating two batches gives an effective batch of roughly 16 without storing all 16 images' activations simultaneously. CUDA automatic mixed precision and gradient scaling are enabled when a GPU is available. The final incomplete accumulation group is normalized by its actual number of batches. Gradients are clipped to norm 1.0. Frequency transforms and attention calculations explicitly use float32 for stability.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 33. Which checkpoint is deployed and why?

The deployment calibrated.pt has the same bytes as results/runs/frequency/best.pt. It is epoch 6, with 600 optimizer steps and the first best balanced validation score of 98.5149%. Epoch 10 ties that score but strict improvement is required, so epoch 6 remains selected. Epoch 11 is the last checkpoint.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 34. What is the difference between resuming and transfer learning?

Resume restores the existing model, optimizer, scaler, epoch, history and saved Torch RNG state, and requires the same manifest hash and architecture. Transfer initialization copies only trained model weights, checks architecture and class order, records parent provenance, and starts with a fresh optimizer and calibration. They are mutually exclusive options.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 35. Was the new test split guaranteed untouched throughout development?

The current review used its existing assignments without further tuning. The supplied records do not establish every earlier use of that split, and generated source IDs do not prove identity separation. Describe it as the existing test split, not a proven external or generator-disjoint benchmark.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Metrics and calibration

### 36. What accuracy does the new model achieve?

The deployed checkpoint reproduces 98.64% accuracy and 98.51% balanced accuracy on 220 validation images. A fresh review measured 93.19% accuracy and 93.12% balanced accuracy on 191 test images. The 95% test goal was not met. These scores evaluate the CNN dataset transform, not the full web-app override policy.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 37. Can the new score be called an improvement over the earlier CNN baseline?

Not as a controlled conclusion. The old baseline scored 93.90% on 2,901 legacy images; the new CNN is evaluated on 191 different images under a different protocol. A fair architectural comparison requires the same data, preprocessing and training budget. No new CNN-only ablation is supplied.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 38. Why report balanced accuracy?

Accuracy counts every correct image equally. Balanced accuracy averages real recall and fake recall. On the current test these are 83/94 and 95/97, giving 93.12%. It helps distinguish class behavior instead of hiding one class behind the other.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 39. How do you explain the new confusion matrix?

The test matrix is [[83,11],[2,95]], with true classes in rows and predicted classes in columns. There are 83 correct real images, 95 correct fake images, 11 real images falsely flagged fake and two missed fake images. That is 178 correct out of 191.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 40. What do precision, recall, F1 and AUC mean for the new test?

Fake precision is the fraction of predicted fakes that are labeled fake: 89.62%. Fake recall is the fraction of labeled fakes detected: 97.94%. Fake F1 is 0.9360. ROC AUC is 0.9947 and measures ranking across thresholds; it is not 99.47% accuracy.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 41. Is the new confidence calibrated?

No. The checkpoint contents say calibrated=False and temperature=1.0. Its filename calibrated.pt is misleading, and the old calibration.json belongs to a different model. Ordinary CNN confidence is an uncalibrated softmax estimate. The external override uses a fixed 99% display, not a fitted confidence estimate.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 42. Why does the app sometimes show exactly 99%?

app.py assigns 0.99 whenever the external detector message signals artificial score >0.20. It is a hard-coded display constant, not the external score and not evidence of 99% correctness. The original CNN probabilities and heatmap class remain in the details.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Code, storage and running

### 43. Which environment was inspected for this revision?

Python 3.14.3, CPU PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.19.0 and Gradio 6.29.1 were observed. CUDA is unavailable in that environment. requirements.txt still restricts Transformers to below 5, so a new installation may differ. The old environment lock is historical.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 44. Which database is used?

No application database or database access layer is implemented. The project uses the filesystem: CSV manifests and retained predictions, JSON configuration/provenance/metrics, PyTorch .pt checkpoints, an ONNX face detector and image files. The UI holds a loaded predictor in memory and Gradio may create temporary upload/output files.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 45. Which code files should you explain?

Explain model.py for CNN/DCT/FGEM/dropout, faces.py for crop/fallback, predict.py for CNN inference, explain.py for Grad-CAM, metadata.py for byte/EXIF checks, ai_detector.py for the external pipeline and threshold, and app.py for seven outputs and the override. forensics.py has ELA/FFT helpers but they are not wired into the app.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 46. How do you launch and demonstrate the current app?

Run Start_App.bat or python src/app.py --open-browser from the root with the intended environment. Confirm local model assets and external-detector availability. Read the actual CNN heatmap class and any external override separately. Test tiny invalid input as well as no-face fallback; the old blank-rejection demo no longer applies.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 47. Do the YAML files define the new experiment?

No YAML loader is implemented. The current checkpoint training_args, source optimizer settings and results/runs/frequency/history.json are the relevant records. Old YAML descriptions and historical run JSON can differ from the new experiment.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Testing and reproducibility

### 48. What verification exists for the updated version?

The review checks model and manifest hashes, all referenced images, validation reproduction, a fresh CNN test evaluation, current Grad-CAM examples, blank fallback and a controlled callback override case. The software-test outcome is recorded in REVIEW_AND_VERIFICATION.md. The historical six-output API verifier is stale and cannot establish a current full-system pass.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 49. What did the controlled override check show?

The exact saved callback was run with a stub CNN REAL result and a stub 21% artificial external message. It returned FAKE (AI Generated) at 99% while retaining the blue REAL heatmap and the original 80%/20% CNN probabilities. This tests control flow, not actual external-model accuracy.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 50. How are the new results linked to the model?

The deployment hash starts 317e1c1e and matches the new best.pt. Its manifest hash starts 9cda41de and matches the existing CSV. Review evidence retains the exact full hashes, metrics and prediction rows. Old evaluation records identify a different model hash beginning 9ea2ddb9.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 51. Do the old confidence intervals apply to the new checkpoint?

No. The old source-group intervals were computed for historical models and splits. The new test has generated per-file group IDs rather than verified source clusters. Do not copy the old intervals or imply cluster independence that the current data cannot establish.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Limitations and future work

### 52. What are the main current limitations?

The new test score is below 95%, the hybrid app has no full-system benchmark, override confidence is fixed, heatmap and final label can disagree, metadata checks are heuristic, split provenance is weak, current confidence is uncalibrated and dependency records disagree. Accepted full-image inputs do not prove broad generator reliability.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 53. Can more datasets be added later, and were the old ones mixed into this run?

More data can be added in a future documented experiment with compatible labels, preprocessing and disjoint splits. The current checkpoint records no transfer parent and its manifest supports only the new sample. The available evidence does not show that the old DFF/FFPP data were mixed into this run.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 54. Which improvements should come first?

Separate the evidence outputs and fix the override/heatmap/confidence mismatch, update seven-output integration tests, make preparation deterministic, record source provenance and calibrate the selected model. Then evaluate any fusion rule on separate data and test larger, independent generator/identity groups.

Evidence: Current source implementation; review evidence; report chapter matching this topic

### 55. How should you summarize the updated project?

We implemented CNN-plus-frequency detection with Grad-CAM, added a new face-image sample experiment and integrated metadata and an external visual detector. The selected CNN has 98.64% validation accuracy and 93.19% on the existing test split. Full-app accuracy and modern-generator generalization remain unverified, and we clearly separate each evidence source.

Evidence: Current source implementation; review evidence; report chapter matching this topic

## Updated application details

### 56. Does checking for SynthID or c2pa text verify a watermark or signature?

No. metadata.py only looks for byte substrings and EXIF fields. It has no invisible-watermark decoder or C2PA signature validator. A matching string or absent EXIF is not proof of generation.

Evidence: src/app.py; src/deepfake/metadata.py; src/deepfake/ai_detector.py; src/prepare_ai_dataset.py

### 57. Are ELA and FFT displayed in the current application?

No. forensics.py defines helpers, but the current app and predictor do not call them. The CNN frequency path uses DCT on learned feature maps. Do not list ELA/FFT as active user features.

Evidence: src/app.py; src/deepfake/metadata.py; src/deepfake/ai_detector.py; src/prepare_ai_dataset.py

### 58. Why can CLI and browser predictions differ?

The CLI calls Predictor alone. The browser additionally runs metadata and the external visual classifier, then may override the displayed class. Its map still comes from Predictor. These are different decision workflows.

Evidence: src/app.py; src/deepfake/metadata.py; src/deepfake/ai_detector.py; src/prepare_ai_dataset.py

### 59. What is the exact external threshold?

The artificial score must be strictly greater than 0.20. A score of 0.20 itself does not trigger the rule. The code then uses a fixed 0.99 confidence display, which has no local calibration evidence.

Evidence: src/app.py; src/deepfake/metadata.py; src/deepfake/ai_detector.py; src/prepare_ai_dataset.py

### 60. Why does the current preparation script fail to reproduce the saved experiment exactly?

It shuffles and assigns splits without setting a data seed and only creates train/val rows. The existing manifest has train/val/test. Preserve the saved manifest and document how its test split was formed before claiming full reproducibility.

Evidence: src/app.py; src/deepfake/metadata.py; src/deepfake/ai_detector.py; src/prepare_ai_dataset.py