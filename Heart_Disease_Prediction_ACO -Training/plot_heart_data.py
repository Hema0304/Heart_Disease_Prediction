import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')

# Define colors (colorblind-friendly)
colors = ['#1f77b4',  # Blue for disease
          '#ff7f0e']  # Orange for no disease

# Age vs Cholesterol
plt.figure(figsize=(10, 6))
for label, color in zip([1, 0], colors):
    subset = df[df['target'] == label]
    plt.scatter(subset['age'], subset['chol'],
                label='Disease' if label == 1 else 'No Disease',
                color=color, alpha=0.7, edgecolor='k')
plt.title('CVD based on Age and Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Age vs Chest Pain Type
plt.figure(figsize=(10, 6))
for label, color in zip([1, 0], colors):
    subset = df[df['target'] == label]
    plt.scatter(subset['age'], subset['cp'],
                label='Disease' if label == 1 else 'No Disease',
                color=color, alpha=0.7, edgecolor='k')
plt.title('CVD based on Age and Chest Pain Type')
plt.xlabel('Age')
plt.ylabel('Chest Pain Type')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
