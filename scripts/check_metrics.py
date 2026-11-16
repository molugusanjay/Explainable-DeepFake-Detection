"""
DeepFake Detection Project - Performance & Evaluation Metrics Viewer
====================================================================
Prints Accuracy, Confusion Matrix, Precision, Recall, F1-Score & ROC-AUC.
"""
import json
from pathlib import Path

def print_metrics():
    root = Path(__file__).resolve().parent.parent
    
    # Paths to metric files
    review_path = root / 'docs' / 'Academic_Submission' / 'Evidence' / 'current_review.json'
    benchmark_path = root / 'models' / 'checkpoints' / 'final_frequency' / 'evaluation' / 'test_metrics.json'
    
    print("\n" + "="*65)
    print("      DEEPFAKE DETECTION - MODEL PERFORMANCE METRICS")
    print("="*65)
    
    if review_path.exists():
        with open(review_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        test = data.get('test', {})
        cm = test.get('confusion_matrix', [[0,0],[0,0]])
        tn, fp = cm[0]
        fn, tp = cm[1]
        
        real_prec = tn / (tn + fn) if (tn + fn) > 0 else 0
        real_rec  = tn / (tn + fp) if (tn + fp) > 0 else 0
        real_f1   = 2 * real_prec * real_rec / (real_prec + real_rec) if (real_prec + real_rec) > 0 else 0
        
        fake_prec = test.get('precision_fake', 0)
        fake_rec  = test.get('recall_fake', 0)
        fake_f1   = test.get('f1_fake', 0)
        
        acc       = test.get('accuracy', 0)
        bal_acc   = test.get('balanced_accuracy', 0)
        auc       = test.get('auc', 0)
        
        print("\n[1] HELD-OUT TEST EVALUATION (191 images)")
        print("-" * 65)
        print(f"  Overall Accuracy    : {acc * 100:.2f}%")
        print(f"  Balanced Accuracy   : {bal_acc * 100:.2f}%")
        print(f"  ROC-AUC Score       : {auc:.4f} ({auc * 100:.2f}%)")
        print("-" * 65)
        
        print("\n  Classification Report:")
        print(f"  {'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
        print("  " + "-"*58)
        print(f"  {'REAL':<12} | {real_prec*100:>8.2f}% | {real_rec*100:>8.2f}% | {real_f1*100:>8.2f}% | {tn+fp:>8}")
        print(f"  {'FAKE':<12} | {fake_prec*100:>8.2f}% | {fake_rec*100:>8.2f}% | {fake_f1*100:>8.2f}% | {fn+tp:>8}")
        print("  " + "-"*58)
        print(f"  {'Total / Avg':<12} | {((real_prec+fake_prec)/2)*100:>8.2f}% | {bal_acc*100:>8.2f}% | {((real_f1+fake_f1)/2)*100:>8.2f}% | {test.get('count', 191):>8}")
        
        print("\n  Confusion Matrix:")
        print("                      Predicted REAL    Predicted FAKE")
        print(f"    Actual REAL           {tn:<14}  {fp:<14} (Total: {tn+fp})")
        print(f"    Actual FAKE           {fn:<14}  {tp:<14} (Total: {fn+tp})")
        print(f"\n    * Correct Predictions : {tn + tp} / {test.get('count', 191)} ({acc * 100:.2f}%)")
        print(f"    * False Positives     : {fp} (Real flagged as Fake)")
        print(f"    * False Negatives     : {fn} (Fake missed as Real)")

        val = data.get('val', {})
        val_acc = val.get('accuracy', 0)
        val_f1 = val.get('f1_fake', 0)
        print("\n" + "-" * 65)
        print(f"[2] VALIDATION SET PERFORMANCE (220 images): Accuracy = {val_acc*100:.2f}%, F1 = {val_f1*100:.2f}%")

    if benchmark_path.exists():
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            bench = json.load(f)
        b_acc = bench.get('accuracy', 0)
        b_prec = bench.get('precision_fake', 0)
        b_rec = bench.get('recall_fake', 0)
        b_f1 = bench.get('f1_fake', 0)
        b_auc = bench.get('auc', 0)
        b_cm = bench.get('confusion_matrix', [[0,0],[0,0]])
        print("\n" + "-" * 65)
        print(f"[3] LARGE MULTI-DOMAIN BENCHMARK (2,901 images across FF++ & DFF):")
        print(f"    Accuracy: {b_acc*100:.2f}% | F1-Score: {b_f1*100:.2f}% | ROC-AUC: {b_auc:.4f}")
        print(f"    Confusion Matrix: [[{b_cm[0][0]}, {b_cm[0][1]}], [{b_cm[1][0]}, {b_cm[1][1]}]]")
        
    print("\n" + "="*65)
    print("  Artifacts & Visual Plots Location:")
    print(f"  * Confusion Matrix Plot: models/checkpoints/final_frequency/evaluation/test_charts.png")
    print(f"  * Academic Project Report: docs/Academic_Submission/Comprehensive_Project_Report.pdf")
    print("="*65 + "\n")

if __name__ == '__main__':
    print_metrics()
