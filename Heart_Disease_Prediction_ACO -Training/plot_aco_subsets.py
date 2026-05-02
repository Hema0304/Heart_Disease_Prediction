import matplotlib.pyplot as plt
import numpy as np

subsets_data = {
    "Subset 1": {
        "DT": [0.984, 0.9769, 0.9939, 0.9849],
        "KNN": [0.8551, 0.8778, 0.8323, 0.8545],
        "RF": [0.9860, 0.9819, 0.9909, 0.9863],
        "XGB": [0.9908, 0.9939, 0.951, 0.9942]
    },
    "Subset 2": {
        "DT": [0.9470, 0.9455, 0.9512, 0.9483],
        "KNN": [0.8692, 0.8720, 0.8720, 0.8720],
        "RF": [0.9611, 0.9633, 0.9604, 0.9618],
        "XGB": [0.9502, 0.9485, 0.9503, 0.9514]
    },
    "Subset 3": {
        "DT": [0.9190, 0.9259, 0.9146, 0.9202],
        "KNN": [0.9097, 0.9116, 0.9116, 0.9116],
        "RF": [0.9221, 0.9344, 0.9116, 0.9228],
        "XGB": [0.9221, 0.9427, 0.9024, 0.9221]
    },
    "Subset 4": {
        "DT": [0.9190, 0.9259, 0.9146, 0.9202],
        "KNN": [0.9097, 0.9116, 0.9116, 0.9116],
        "RF": [0.9221, 0.9344, 0.9116, 0.9228],
        "XGB": [0.9221, 0.9427, 0.9024, 0.9221]
    },
    "Subset 5": {
        "DT": [0.9252, 0.9222, 0.9299, 0.9271],
        "KNN": [0.8411, 0.8531, 0.8323, 0.8426],
        "RF": [0.9283, 0.9222, 0.9390, 0.9305],
        "XGB": [0.9159, 0.9006, 0.9390, 0.9194]
    },
    "Subset 6": {
        "DT": [0.9408, 0.9290, 0.9573, 0.9429],
        "KNN": [0.8769, 0.8903, 0.8659, 0.8779],
        "RF": [0.9377, 0.9286, 0.9512, 0.9398],
        "XGB": [0.9003, 0.9099, 0.8933, 0.9015]
    }
}

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
models = ['DT', 'KNN', 'RF', 'XGB']

# Blue-themed colors from light to dark
model_colors = {
    'DT': '#99ccff',  # Light Blue
    'KNN': '#66b3ff', # Soft Blue
    'RF': '#3399ff',  # Medium Blue
    'XGB': '#0066cc'  # Deep Blue
}

for subset_name, data in subsets_data.items():
    values = np.array([data[model] for model in models])
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(metrics))
    for i, model in enumerate(models):
        ax.bar(index + i * bar_width, values[i], bar_width, 
               label=model, color=model_colors[model])
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(f'Model Performance - {subset_name}')
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.75, 1.05)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
