import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import missingno as msno
import itertools

os.makedirs('results', exist_ok=True)

df = pd.read_csv('heart.csv')

selected_features = ['chol', 'restecg', 'oldpeak', 'ca']
df_selected = df[selected_features]

plt.figure(figsize=(10, 6))
corr = df_selected.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap (Selected Features 4,6,9,11)')
plt.tight_layout()
plt.savefig('results/correlation_heatmap_selected_4_6_9_11.png')
plt.close()

msno.matrix(df_selected)
plt.title('Missing Value Heatmap (Selected Features 4,6,9,11)')
plt.tight_layout()
plt.savefig('results/missing_value_heatmap_selected_4_6_9_11.png')
plt.close()

sns.clustermap(corr, annot=True, cmap='viridis', linewidths=0.5)
plt.savefig('results/cluster_heatmap_selected_4_6_9_11.png')
plt.close()

continuous_features = ['chol', 'oldpeak']

for x_var, y_var in itertools.combinations(continuous_features, 2):
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        x=df_selected[x_var],
        y=df_selected[y_var],
        cmap='mako',
        fill=True,
        thresh=0.05
    )
    plt.title(f'Density Heatmap: {x_var} vs {y_var} (Selected Features)')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.tight_layout()
    plt.savefig(f'results/density_heatmap_{x_var}_vs_{y_var}_selected.png')
    plt.close()

print("Heatmaps for features 4, 6, 9, 11 generated and saved in the 'results' folder.")
