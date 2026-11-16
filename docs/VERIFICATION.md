# Current verification record

Review date: 8 October 2026. Scope: the source code, configuration, documentation, retained experiment records, final model and examples in DeepFake_Detection_FYP.

## Software and documentation corrections

- Corrected the Windows launcher working directory and quoted paths, added a checkpoint guard and recognized the existing sibling Python environment.
- Corrected relocated helper roots, script/module paths and import bootstrapping. Relative downloader paths resolve before changing directory.
- Made historical orchestration help/import behavior safe and added four relocation regression checks. No experiment was started by this review.
- Corrected README architecture, temperature, subgroup metrics and unsupported claims. YAML files are marked as descriptive records rather than an implemented configuration loader.
- Rewrote local setup, reproduction and data-location instructions to match this folder. Clarified missing training prerequisites and corrected prediction output help.
- Restored the retained YuNet license beside its detector. Updated report and viva entry points to the reviewed materials.

The trained weights, class order, preprocessing, decision threshold and measured experiment scores were not changed.

## Checks completed

The project Python environment ran `python -m pytest -q`: **20 passed in 82.28 seconds**. This includes numerical DCT and gradient checks, Grad-CAM class/zero-map behavior, data integrity, balancing, transfer checks and four relocation checks. This result checks software behavior; it does not estimate model accuracy.

A fresh `python scripts/verify_app.py` run exercised the local upload API. It returned a REAL example at 97.9% and a FAKE example at 98.0%, with heatmaps, validation-calibrated confidence, blank-input rejection and stale-output clearing. The temporary server closed after the check. The saved record is `models/checkpoints/final_frequency/app_verification.json`. This was an API check, not a completed visual browser review.

All retained final prediction rows were audited. Recomputed accuracy, balanced accuracy, AUC, confusion matrices and source-cluster intervals matched the saved final evaluation records. The deployment checkpoint SHA-256 matched both evaluation records:

`9ea2ddb91ebf833d658252336161935a4d3d6634823fb871a6ce214282d01aa2`

## Measured results

| Evaluation | Accuracy | Balanced accuracy | ROC AUC | Accepted images |
|---|---:|---:|---:|---:|
| Final frequency, combined legacy test | 92.66% | 89.18% | 0.9646 | 2,901 |
| Final frequency, additional DFF | 89.24% | 91.04% | 0.9792 | 1,320 |
| CNN-only, combined legacy test | 93.90% | 90.09% | 0.9725 | 2,901 |

The 95% point-estimate accuracy target was not met. CNN-only values are supported by the retained metrics JSON; its full checkpoint, predictions and history are absent from this folder. Frequency enhancement did not improve combined legacy accuracy in the recorded comparison.

## Document verification

The final report has 33 pages, the viva guide 15 pages and 55 questions, and the explanation guidebook 19 pages. Each PDF includes a linked contents page and bookmarks. Rendered pages were visually inspected. The final PDFs have no empty pages or text outside page bounds in the automated geometry check. Mathematical exponents, page breaks, result labels and captions were checked.

The presentation has 34 slides with speaker notes, five native charts with embedded data workbooks, seven native tables and editable process diagrams. All slide renders were inspected, including revised chart labels. Package integrity, geometry, font consistency and chart/workbook checks passed. This does not claim that the presentation was opened in native Microsoft PowerPoint.

The heatmap pairs are actual saved project outputs. They include a correct FAKE example, a correct REAL example in the slides, and a Face2Face error predicted REAL at about 95.7%. These illustrate class relevance, not a validated pixel-level forgery mask. No fabricated UI screenshot or replacement heatmap was used.

## Review boundaries and remaining presentation preparation

This reorganized folder lacks raw training data, complete preparation manifests, frozen holdout selection records and some historical checkpoints. Recorded training counts and historical data separation cannot be independently reconstructed from this copy alone. Legacy tests were previously inspected; the additional DFF set uses known methods and does not establish identity-disjoint or unseen-generator generalization. No new training, external robustness evaluation or pixel-level localization evaluation was performed.

Add the institution's required student names, roll numbers, college, department and supervisor information before formal submission. These were not supplied and were not invented. Rehearse the live app and slide show on the presentation computer. Follow institutional requirements for acknowledging implementation assistance.

The accompanying ZIP is a documentation pack, not a full model-and-dataset archive. Editable Markdown companions contain the written content; diagrams and images are embedded in the PDFs and presentation. `MANIFEST.json` records final document hashes. Previous versions and build intermediates remain outside this submission pack.
