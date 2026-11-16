# Review and verification — 9 October 2026

The four academic documents describe the current implementation. The review did not change application code, datasets or model weights, and did not train or recalibrate the model.

## Current evidence

- Deployment SHA-256: 317e1c1e9a73eaeff6d1454ec5e926775597ad23e612e390eda99d4d3f7f1140.
- Deployment matches results/runs/frequency/best.pt, selected epoch 6. Temperature 1.0; calibrated=false.
- All 2,000 manifest images exist and match their stored hashes. The current preparation script does not reproduce the saved three-way split.
- Validation: 220 images, 98.64% accuracy, 98.51% balanced accuracy.
- Existing test: 191 images, 93.19% accuracy, 93.12% balanced accuracy. Matrix [[83,11],[2,95]], AUC 0.9947.
- Evaluation used the custom CNN with direct dataset resizing and normalization. It did not measure the full face-crop-plus-external-override web application. No training, calibration or threshold tuning occurred during evaluation.
- The existing test split is not proven never previously inspected, identity-disjoint or generator-disjoint.
- 20 software tests passed in 54.78 seconds. See Evidence/pytest_final.txt. This suite does not establish full hybrid-app correctness.
- Current REAL/FAKE examples and a blank-image full-frame fallback were exercised through Predictor.
- A controlled callback with stubbed CNN/external results confirms the FAKE/99% override retains CNN probabilities and the CNN-class heatmap. This is a control-flow check, not an external-model benchmark.

## Material issues documented

The external score threshold is strictly greater than 0.20. The displayed 99% is a constant. A FAKE override can retain a blue REAL heatmap. Metadata inspection is byte/EXIF matching, not verified C2PA or SynthID. ELA/FFT helpers are not called by current inference. Historical calibration, results and six-output API records do not describe the new deployment. Installed Transformers 5.19.0 conflicts with the requirements range below 5. These issues are reported, not silently repaired as part of a document-only revision.

## Artifact checks

The 34-slide PPTX passed package, geometry, font, native-table and embedded-chart checks, and imported successfully into the artifact renderer. Slides were rendered and visually inspected; an overflow on the conclusion slide was corrected. The PDFs were rendered with Poppler, checked visually, and their text and contents pages inspected. Microsoft PowerPoint itself was not opened, and a clean installation or complete live hybrid-app benchmark was not performed.

There are 33 report pages, 16 viva-guide pages with 60 questions, and 16 guidebook pages. Add verified student, supervisor and institution details in the required academic format before submission.

## Evidence files

Evidence/current_review.json records current metrics, examples and the explicitly labelled stubbed callback check. Evidence/review_predictions.csv contains the 220 validation and 191 test predictions. Evidence/checkpoint_audit.json records checkpoint and manifest metadata. Source hashes bind the review to the inspected code. Current example originals and heatmaps are included for demonstration. Some raw Predictor text still says oval; the documentation explicitly corrects that stale description.

Previous deliverables are backed up under the project's .build/revision_20261009/previous_materials directory. Matching revised copies are supplied in project docs/Academic_Submission and the user's kits project/Academic_Submission folder. The ZIP contains the academic package, not the full training data or model.
