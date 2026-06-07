"""Threshold tuning, ROC plotting, and SHAP feature-name helpers for the
final evaluation notebook."""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def find_best_threshold_f2(y_test, probs):
    """Find the decision threshold that maximises the F2 score (recall-weighted).

    For a readmission-risk model, missing a true readmission (false negative)
    is costlier than a false alarm, so we weight recall more heavily than
    precision when choosing the operating threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f2 = (5 * precisions * recalls) / (3 * precisions + recalls + 1e-9)
    best_thresh = thresholds[f2.argmax()]
    return best_thresh


def plot_roc_curve(y_test, probs, threshold, model_label="Model", save_path=None):
    """Plot the ROC curve, marking the chosen operating threshold."""
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#378ADD", lw=2, label=f"{model_label} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.500)")
    plt.scatter(*[(fpr[thresholds >= threshold][-1]), (tpr[thresholds >= threshold][-1])],
                color="#D85A30", zorder=5, s=80, label=f"Threshold = {threshold:.3f}")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve — {model_label}")
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def get_feature_names_out(preprocessor, cat_cols):
    """Reconstruct the post-transform feature names from a fitted ColumnTransformer.

    Needed to label SHAP plots — the transformer flattens numeric, one-hot
    encoded categorical, and ordinal columns into a single feature matrix.
    """
    num_feature_names = preprocessor.named_transformers_["num"].get_feature_names_out().tolist()
    ord_feature_names = preprocessor.named_transformers_["ord"].get_feature_names_out().tolist()
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    return num_feature_names + cat_feature_names + ord_feature_names
