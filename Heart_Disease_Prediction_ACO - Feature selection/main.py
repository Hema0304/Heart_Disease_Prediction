import os
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from src.utils import (
    load_data, load_selected_features,
    evaluate_model, plot_confusion_matrix,
    plot_combined_roc, plot_performance_comparison
)
from src.models import decision_tree, knn, random_forest, xgboost_model

def main():
    X_df, X_scaled, y = load_data("data/heart.csv")
    selected_subsets = load_selected_features("results/selected_features.txt")

    all_performance = []
    all_roc_data = []

    models = {
        "decision_tree": decision_tree.get_model(),
        "knn": knn.get_model(),
        "random_forest": random_forest.get_model(),
        "xgboost": xgboost_model.get_model()
    }

    for subset_idx, subset in enumerate(selected_subsets):
        print(f"\nEvaluating Subset {subset_idx+1}: {subset}")
        X_selected = X_scaled[:, subset]
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, stratify=y, random_state=42
        )

        subset_results = {}
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
            subset_results[model_name] = metrics

            plot_confusion_matrix(y_test, metrics["predictions"], model_name, subset_idx)

            fpr, tpr, _ = roc_curve(y_test, metrics["probabilities"])
            all_roc_data.append((f"{model_name} (Subset {subset_idx+1})", fpr, tpr))

        all_performance.append(subset_results)

    plot_combined_roc(all_roc_data)
    plot_performance_comparison(all_performance)
    print("\nAll results saved in 'results/' folder.")

if __name__ == "__main__":
    os.makedirs("results/confusion_matrices", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)
    main()
