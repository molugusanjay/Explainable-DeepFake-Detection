# Architecture and relationship to the base paper

Base: Zhou et al., **MH-FFNet: Leveraging mid-high frequency information for robust fine-grained face forgery detection**, Expert Systems with Applications 276 (2025), 127108. DOI: https://doi.org/10.1016/j.eswa.2025.127108.

This is an independent, paper-inspired implementation, not the authors' official code and not a claim of exact reproduction.

## Retained ideas

1. A ConvNeXt convolutional backbone extracts spatial features from a 224 x 224 face crop.
2. P-FGEM processes nonoverlapping patches of shallow features with DCT, removes low frequencies, reconstructs with inverse DCT, and fuses with the original features through channel attention.
3. F-FGEM performs the same frequency filtering globally on each mid-level feature channel before attention fusion.
4. Remaining CNN stages produce two logits, in the fixed order REAL=0, FAKE=1.
5. Grad-CAM differentiates the predicted-class logit through the complete spatial/frequency model. The final convolutional stage supplies the activations.

## Explicit implementation choices and changes

| Item | Paper | This package |
|---|---|---|
| Backbone | ImageNet-pretrained ConvNeXt-Base | ConvNeXt-Tiny by default for free GPU training; Base selectable |
| Face detection | MTCNN | OpenCV YuNet; identical crop policy for training and inference |
| Frequency patches/filter | DCT with low-frequency removal; precise filter cutoff and patch size not specified in the supplied methodology | 7 x 7 feature patches; mask retains (u/(H-1)+v/(W-1))/2 > 0.25 |
| DCT normalization | Equations omit normalization factors | Orthonormal DCT-II and its exact inverse |
| Attention | Description emphasizes channel interaction; equation notation has inconsistencies | C x C attention: softmax(Q K^T / sqrt(C H W)) V, residual sum, 1 x 1 projection |
| Optimization | AdamW, LR 4e-5, batch 16, 20 epochs | Same LR/optimizer; batch 8, accumulation 2; six-epoch mixed comparison, patience 3 (earlier runs use their recorded budgets) |
| Checkpoint choice | Best validation accuracy | Mean of per-dataset validation balanced accuracies in the mixed run |
| Dataset | FF++ video benchmark | Mixed DeepFakeFace and FF++ C23 subsets; FF++ retains 12 train / 16 evaluation frames per video |
| Dataset size | Full benchmark protocol | Explicit, reproducible per-split subset by default; size recorded |
| Training balance | Not specified here | Equal dataset and class mass, then equal fake-method mass within each dataset |

The mixed experiment includes a same-budget CNN-only comparison. A smaller/subsampled run is not directly comparable to the paper's results. Paper figures (97.37% C23 accuracy and 99.44% C23 AUC) are reference results only. The reviewed final results are 92.66% combined legacy accuracy and 89.24% additional DFF accuracy. See Academic_Submission/Comprehensive_Project_Report.pdf for evidence boundaries and the CNN-only comparison.

## Application and explanation additions

- Single-face upload interface with real/fake prediction and estimated probability.
- Validation-only scalar temperature calibration. This reuses the validation split used for selection; legacy tests were previously inspected; an additional unused-source DFF holdout is reported separately.
- Face-coordinate Grad-CAM overlay; an approximate facial oval limits the display. This is NOT a segmentation model or a verified forgery mask. The restriction can hide attention outside the oval; classification still uses the full crop.
- Red corresponds to positive FAKE-class attention; blue corresponds to positive REAL-class attention. Zero/negative activation never produces fabricated colored marks.
- Text names the strongest broad spatial regions in the displayed heatmap. It does not diagnose particular texture defects or give unsupported causal claims.
- Missing faces, multiple faces and very small faces are rejected. Low confidence gets a visible uncertainty note; the 70% note threshold is a user-interface heuristic, not a validated rejection rule.
- Grad-CAM already appears in the base paper. The interface, calibration and explanation presentation are engineering extensions, not claims of new research novelty.

## Scope and limits

FF++ covers facial manipulation methods, not all modern fully generated faces. Training on FF++ alone does not demonstrate detection of arbitrary diffusion/GAN imagery. A separate, properly sourced synthetic-face dataset and generator-disjoint evaluation would be needed for that broader claim.

The current DeepFakeFace experiment groups the publisher's source/target pairs and exact duplicate images before splitting. This custom subset is not identity-disjoint, not generator-disjoint, and not directly comparable to the paper's FF++ result. The separate FF++ pipeline uses official source-video splits, which also do not prove identity separation. The loader rejects cross-split source IDs, duplicate content hashes, duplicate paths and changed content. Exact hashes do not detect all near-duplicates.

Face-detection failures are excluded from frame metrics and counted in preparation.json. Report those counts; do not present the measured accuracy as performance over all input images. Heatmaps are coarse associations, not localization ground truth.

## References and dependencies

- Base paper: https://doi.org/10.1016/j.eswa.2025.127108
- Current DeepFakeFace dataset: https://huggingface.co/datasets/OpenRL/DeepFakeFace
- Publisher generation and pairing code: https://github.com/OpenRL-Lab/DeepFakeFace
- Official FF++ dataset, access instructions and splits: https://github.com/ondyari/FaceForensics
- ConvNeXt implementation: https://docs.pytorch.org/vision/stable/models/convnext.html
- YuNet face detector and its licensing: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- Grad-CAM paper: https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Colab free resource limits: https://research.google.com/colaboratory/faq.html

Datasets and pretrained weights remain subject to their original terms. This folder includes the trained detector and YuNet. The YuNet license is stored at models/assets/YuNet-LICENSE.txt. Dataset images and private access scripts are excluded.
