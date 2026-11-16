# Reproduction and verification

Run from the `DeepFake_Detection_FYP` root in its Python environment. On this laptop, replace `python` with `& '..\work\venv\Scripts\python.exe'` in PowerShell. The supplied checkpoint supports inference. This reorganized copy lacks raw datasets, complete preparation manifests and historical parent checkpoints, so it is not a self-contained archive of every training experiment.

## Demonstrate the existing model

```powershell
python src/app.py --open-browser
python src/predict.py models/checkpoints/final_frequency/demo/2_original.png --output results/classroom_demo
python -m pytest -q
python scripts/verify_app.py
```

The last command starts a temporary localhost server, exercises REAL/FAKE uploads and invalid-input clearing, then closes it. It does not replace a visual browser rehearsal. `scripts/verify_final.py` needs original dataset files that are absent from this copy.

## Restore data before a new experiment

DeepFakeFace uses publisher archives and actual pairing metadata:

```powershell
python scripts/download_dff.py
python scripts/prepare_dff.py --groups 1500 --seed 42
```

For FaceForensics++, obtain approved access and retain its downloader privately. The wrapper accepts the local script path. The access email specified EU2:

```powershell
python scripts/run_local.py download --dataset ffpp --downloader "C:\path\to\approved_downloader.py" --server EU2
python scripts/prepare_ffpp.py --videos data/ffpp_videos --output data/prepared
```

Replace the example path with the approved script's actual path. The wrapper's default is 300 C23 videos per category across five categories. The downloader's terms prompt remains interactive. There is no `Download_FaceForensics.cmd` in this reorganized package. Never redistribute private access material.

Data paths and recorded split counts appear in `data/README.md`. Exact counts depend on source files and detection outcomes. Recorded counts alone do not reproduce manifests or prove source separation.

## Controlled mixed comparison

After both prepared manifests and their image files are restored and verified:

```powershell
python src/run_mixed_experiment.py
```

The runner uses `data/mixed_manifest.csv` and writes under `results/runs/mixed_comparison`. It compares ImageNet-initialized ConvNeXt-Tiny with and without frequency modules, using seed 42, AdamW at 4e-5, batch 8, accumulation 2, a six-epoch maximum and patience 3. Selection averages dataset validation balanced accuracies. YAML files describe settings; the runner does not parse them as executable configuration.

Inspect output paths and existing run state before starting a new protocol. The runner resumes its own interrupted stages and retains existing evaluations. Use separate output directories for a genuinely new experiment and preserve the submitted evidence. This documentation review did not train or alter weights.

## Historical refinement and additional holdout

`scripts/prepare_fresh_holdout.py` selects unused DFF source pairs and requires the original DFF archives and both prepared manifests. A frozen existing holdout must not be regenerated after inspecting results. Its recorded selection uses 300 unused pairs and seed 20261007.

`src/refine_final_model.py` describes a bounded four-epoch refinement from the earlier FF++ transfer checkpoint. That parent and complete mixed runs are absent from this package. Restore and verify those prerequisites before attempting historical reproduction. `src/evaluate_fresh_holdout.py` also requires the prepared fresh holdout and expected experiment checkpoints. Do not run these commands expecting the inference checkpoint alone to supply the missing inputs.

The selected deployment checkpoint is `models/checkpoints/final_frequency/calibrated.pt`. The saved selection retained the mixed frequency candidate. Legacy tests had already been inspected. The additional DFF evaluation used unused source pairs but known methods; no fresh FF++ holdout is documented. Source grouping does not establish identity or generator separation.

## Interpretation and evidence

The reviewed report in `docs/Academic_Submission` distinguishes independently recomputed final prediction metrics from recorded historical protocol and CNN metrics-only evidence. The 95% accuracy goal was not met. Grad-CAM is class relevance, not a verified edited-pixel mask. Validation-only temperature fitting reuses checkpoint-selection data and may be optimistic on new distributions.

Package and numerical differences can affect repeatability. Consult the saved checkpoint metadata, histories, calibration, selection and per-image prediction files. `docs/Academic_Submission/REVIEW_AND_VERIFICATION.md` records the review's checks and boundaries.
