# DeepFakeFace dataset record

This is a custom source-pair-disjoint subset, not an official benchmark split or an identity-disjoint experiment.

Source: https://huggingface.co/datasets/OpenRL/DeepFakeFace
Publisher pairing code: https://github.com/OpenRL-Lab/DeepFakeFace

Release revision: `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`. Selection seed: 42. Selected source pairs: 1,500.

| Split | Real Wiki | Inpainting | InsightFace | Text-to-image | Total | Source groups |
|---|---:|---:|---:|---:|---:|---:|
| train | 1154 | 1296 | 1223 | 1480 | 5153 | 1195 |
| val | 133 | 146 | 131 | 177 | 587 | 149 |
| test | 140 | 159 | 147 | 183 | 629 | 150 |

8,100 candidate images were selected from the four 30,000-image archives; 6,369 passed the common single-face crop policy and 1,731 were rejected (21.37%). Rejections are recorded by method and reason in `data/dff_prepared/preparation.json`.

Real and fake classes are not equally frequent. Training uses inverse-class-frequency sampling; validation and test preserve their selected distributions. Report balanced accuracy alongside ordinary accuracy.

Source/target pairs from the publisher are grouped because an InsightFace image contains a source identity and a destination background. SHA-256 checks verify all four downloads. Raw-file, decoded-pixel and resized-crop exact duplicates are joined before split assignment. Crops are resized to 224 x 224. Validation and testing receive no random augmentation.

Limitations: repeated celebrity identities and near-duplicate photographs can remain across splits. These generators and this image collection differ from future internet uploads. Face-detection failures are excluded from classification accuracy. The full-dataset archives are not redistributed in the project package.
