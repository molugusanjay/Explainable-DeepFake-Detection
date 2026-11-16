## Actual final results

| Evaluation | Accuracy | Balanced accuracy | Images |
|---|---:|---:|---:|
| Combined legacy test | 92.66% | 89.18% | 2901 |
| DFF within legacy test | 89.51% | 90.70% | 629 |
| FF++ within legacy test | 93.53% | 88.56% | 2272 |
| Additional unused-source DFF holdout | 89.24% | 91.04% | 1320 |

The 95% accuracy target is **not met** on the combined legacy test and **not met** on the additional DFF holdout. These statements concern the measured point estimates, not universal detection accuracy.

Legacy test sets had already been inspected before the mixed experiment. The additional holdout uses new DFF source pairs but known manipulation methods. Neither split establishes identity-disjoint or unseen-generator generalization. See Academic_Submission/Comprehensive_Project_Report.pdf for the CNN-only comparison and limitations.

