import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_training_history(history, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss", color="blue")
    axes[0].plot(history.history["val_loss"], label="Val Loss", color="orange")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"], label="Train Acc", color="green")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", color="red")
    axes[1].set_title(f"{title_prefix} Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", cmap="Blues"):
    plt.figure(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False,
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.show()


def plot_roc_curve(y_true_onehot, y_pred_probs, class_names, title="ROC Curve"):
    n_classes = len(class_names)
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
