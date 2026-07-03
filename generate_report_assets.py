import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Confusion Matrix
    cm = np.array([[9, 2], [5, 0]])
    classes = ['Normal', 'Pneumonia']
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, 
                annot_kws={"size": 16, "weight": "bold"},
                cbar_kws={'label': 'Number of Images'})
    
    plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix (Evaluation on Synthetic Validation Set)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    
    cm_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix chart to {cm_path}")

    # 2. Metrics Bar Chart
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Precision', 'Recall', 'F1-Score'],
        'Class': ['Normal', 'Normal', 'Normal', 'Pneumonia', 'Pneumonia', 'Pneumonia'],
        'Value': [0.64, 0.82, 0.72, 0.00, 0.00, 0.00]
    }
    df = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x='Metric', y='Value', hue='Class', data=df, palette='Set2')
    
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.xlabel('Evaluation Metric', fontsize=12, fontweight='bold')
    plt.title('Evaluation Metrics comparison by Class', fontsize=13, fontweight='bold', pad=15)
    
    # Add values on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    plt.legend(title='Class')
    plt.tight_layout()
    
    metrics_path = os.path.join(base_dir, "metrics_chart.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Saved metrics bar chart to {metrics_path}")

if __name__ == "__main__":
    main()
