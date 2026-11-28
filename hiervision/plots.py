# Plotting utilities for the HierVision app.
# Provides matplotlib figures for evaluation metrics like confusion matrices.

import matplotlib.pyplot as plt
import numpy as np

def confusion_figure(matrix: list[list[int]], classes: list[str], title: str, normalize: bool = False) -> plt.Figure:
    # Renders a confusion matrix heatmap using matplotlib to avoid heavier dependencies like seaborn.
    mat = np.array(matrix, dtype=float)
    if normalize:
        # Divide each row by its sum to show proportions instead of absolute counts.
        row_sums = mat.sum(axis=1, keepdims=True)
        # Avoid division by zero warnings
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(mat, cmap="Blues")
    fig.colorbar(cax)

    # Set up ticks and labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left")
    ax.set_yticklabels(classes)

    # Label axes and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=20)

    # Loop over data dimensions and create text annotations to make values readable.
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = mat[i, j]
            text_val = f"{val:.2f}" if normalize else str(int(val))
            # Use white text on dark backgrounds for contrast
            color = "white" if (mat[i, j] > (mat.max() / 2.)) else "black"
            ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return fig