import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def evaluate_subset(X, y, subset):
    X_selected = X[:, subset]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    f1_scores = []
    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        models = [
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            KNeighborsClassifier(),
            LogisticRegression(max_iter=500)
        ]

        f1_model_scores = []
        for model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1_model_scores.append(f1_score(y_test, preds))
        
        f1_scores.append(np.mean(f1_model_scores))

    return np.mean(f1_scores)

def run_aco(X, y, n_ants=50, n_iterations=100, n_subsets=6):
    n_features = X.shape[1]
    subsets = []
    pheromone_trails = []

    for i in range(n_subsets):
        print(f"\nRunning ACO for Subset {i+1}/{n_subsets}")
        pheromone = np.ones(n_features)
        decay = 0.1 + (i * 0.1)
        best_subset = None
        best_score = 0

        for iteration in range(n_iterations):
            all_subsets = []
            all_scores = []

            for ant in range(n_ants):
                prob = pheromone / pheromone.sum()
                selected = np.where(np.random.rand(n_features) < prob)[0]
                if len(selected) < 2:
                    continue

                score = evaluate_subset(X, y, selected)
                all_subsets.append(selected)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_subset = selected

            pheromone *= (1 - decay)
            for idx, subset in enumerate(all_subsets):
                for f in subset:
                    pheromone[f] += all_scores[idx]

            print(f"Iteration {iteration+1}/{n_iterations} - Best F1: {best_score:.4f}")

        mutual_info = mutual_info_classif(X, y)
        top_features = np.argsort(mutual_info[best_subset])[::-1][:min(len(best_subset), 10)]
        final_subset = best_subset[top_features]
        subsets.append(sorted(list(final_subset)))
        pheromone_trails.append(pheromone.copy())

    return subsets

def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("data/heart.csv")

    X = df.drop("target", axis=1).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    subsets = run_aco(X, y)
    with open("results/selected_features.txt", "w") as f:
        for i, subset in enumerate(subsets, 1):
            f.write(f"Subset {i}: {subset}\n")

if __name__ == "__main__":
    main()
