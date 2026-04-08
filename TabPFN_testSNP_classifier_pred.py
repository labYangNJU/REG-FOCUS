import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, average_precision_score, roc_curve, precision_recall_curve
import joblib
#import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description="Predict AISNP using trained TabPFN model")
parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
args = parser.parse_args()

# Step 1: Load the trained model
model = joblib.load('./tabpfn_model_classifier.pkl')
print("Model loaded successfully!")

# Step 2: Load new data
df_new = pd.read_csv(args.input_file)

# Preprocessing: remove ID column; separate X and y (if label exists)
X_new = df_new.drop(['label', 'ID'], axis=1, errors='ignore')
y_new = df_new['label'] if 'label' in df_new.columns else None

print(f"Input data shape: {X_new.shape}")
if y_new is not None:
    print(f"Label distribution: {pd.Series(y_new).value_counts().sort_index()}")

# Step 3: Prediction
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Add prediction results to the DataFrame (optional, for saving results)
df_new['predicted_label'] = predictions
df_new['predicted_proba'] = [prob[predictions[i]] for i, prob in enumerate(probabilities)]  # highest probability

# Record probability for each class (dynamically add columns such as proba_class0, proba_class1, etc.)
n_classes = probabilities.shape[1]
for cls in range(n_classes):
    df_new[f'proba_class{cls}'] = probabilities[:, cls]

print("\nExample of the first 5 prediction results:")
cols = ['predicted_label', 'predicted_proba']
if 'ID' in df_new.columns:
    cols.insert(0, 'ID')
if y_new is not None:
    cols.insert(1 if 'ID' in df_new.columns else 0, 'label')
print(df_new[cols].head())

# Step 4: Calculate evaluation metrics (if true labels are available)
if y_new is not None:
    n_classes = len(np.unique(y_new))
    print("\n=== Overall Performance Metrics ===")
    print(f"Test Accuracy: {accuracy_score(y_new, predictions):.4f}")
    print(f"Test F1 (macro): {f1_score(y_new, predictions, average='macro'):.4f}")

    def compute_auc(y_true, y_proba, n_classes):
        if n_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            return roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

    auc_score = compute_auc(y_new, probabilities, n_classes)
    print(f"Test AUC (ovr-macro): {auc_score:.4f}")

    # Per-class report
    print("\n=== Per-Class Precision/Recall/F1 ===")
    report = classification_report(y_new, predictions, output_dict=True)
    class_names = sorted(np.unique(y_new))
    class_data = []
    for cls in class_names:
        cls_str = str(cls)
        class_data.append({
            'Class': cls_str,
            'Precision': report[cls_str]['precision'],
            'Recall': report[cls_str]['recall'],
            'F1': report[cls_str]['f1-score']
        })

    class_data.append({
        'Class': 'Macro Avg',
        'Precision': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1': report['macro avg']['f1-score']
    })

    class_df = pd.DataFrame(class_data).round(4)
    print(class_df)

    # Pairwise AUROC and AUPR calculation for all class pairs
    print("\nComputing pairwise AUROC and AUPR...")
    pairwise_results = []
    all_pairs = [(0,1), (0,2), (1,2)]  # all class pairs

    for a, b in all_pairs:
        mask = np.isin(y_new, [a, b])
        y_true_ab = np.where(y_new[mask] == a, 1, 0)
        y_score_ab = probabilities[mask, a] - probabilities[mask, b]

        auroc = roc_auc_score(y_true_ab, y_score_ab)
        aupr = average_precision_score(y_true_ab, y_score_ab)

        pairwise_results.append({'Pair': f'{a} vs {b}', 'AUROC': auroc, 'AUPR': aupr})

        # Plot ROC curve
        #fpr, tpr, _ = roc_curve(y_true_ab, y_score_ab)
        #plt.figure(figsize=(6, 4))
        #plt.plot(fpr, tpr, label=f'AUROC={auroc:.4f}')
        #plt.plot([0, 1], [0, 1], 'k--', label='Random')
        #plt.xlabel('False Positive Rate (FPR)')
        #plt.ylabel('True Positive Rate (TPR)')
        #plt.title(f'ROC Curve: Class {a} vs {b}')
        #plt.legend()
        #plt.savefig(f'roc_{a}_vs_{b}.png', dpi=300, bbox_inches='tight')
        #plt.close()

        # Plot PR curve
        #precision, recall, _ = precision_recall_curve(y_true_ab, y_score_ab)
        #plt.figure(figsize=(6, 4))
        #plt.plot(recall, precision, label=f'AUPR={aupr:.4f}')
        #plt.xlabel('Recall')
        #plt.ylabel('Precision')
        #plt.title(f'PR Curve: Class {a} vs {b}')
        #plt.legend()
        #plt.savefig(f'pr_{a}_vs_{b}.png', dpi=300, bbox_inches='tight')
        #plt.close()

        print(f"Pair {a} vs {b}: AUROC={auroc:.4f}, AUPR={aupr:.4f}")

    # Pairwise results table
    pairwise_df = pd.DataFrame(pairwise_results).round(4)
    print("\nPairwise AUROC and AUPR results:")
    print(pairwise_df)

else:
    print("\nNo ground-truth labels provided. Only prediction results will be generated.")

# Step 5: Save results (optional)
input_basename = os.path.basename(args.input_file)

if input_basename.endswith('.csv'):
    output_filename = input_basename[:-4] + '_prediction.csv'
else:
    output_filename = input_basename + '_prediction.csv'

df_new.to_csv(output_filename, index=False)
print(f"\nPrediction results saved to '{output_filename}'.")