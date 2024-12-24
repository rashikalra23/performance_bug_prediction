import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results_file = '/content/drive/MyDrive/bug_pred_results.csv'
results_df = pd.read_csv(results_file)

# Set Seaborn style
sns.set(style="whitegrid")

# Visualization functions

def plot_best_algorithm():
    best_algos = results_df.groupby('algorithm_metric')[['auc', 'mcc', 'f1']].mean().reset_index()
    best_algos = best_algos.melt('algorithm_metric', var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=best_algos, x='algorithm_metric', y='Score', hue='Metric', palette='coolwarm')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.title('Average Performance by Algorithm')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.xlabel("Algorithm and Metric Type")
    plt.tight_layout()
    plt.show()

def plot_best_metric():
    best_metrics = results_df.melt(
        id_vars=['project', 'algorithm_metric'],
        value_vars=['auc', 'mcc', 'f1'],
        var_name='Metric', value_name='Score'
    )
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=best_metrics, x='Metric', y='Score', palette='Set3', inner="quartile")
    plt.title('Evaluation Metric Distribution')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.tight_layout()
    plt.show()

# Generate and display visualizations
plot_best_algorithm()
plot_best_metric()

# Save visualizations as images for the report
plt.savefig('/content/drive/MyDrive/bug_pred_visualizations_best.png')
print("Visualizations saved to /content/drive/MyDrive/bug_pred_visualizations_best.png")
