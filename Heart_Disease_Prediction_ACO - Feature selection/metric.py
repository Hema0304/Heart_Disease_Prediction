import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

df = pd.read_csv("data/heart.csv")
target_col = 'target'
X_full = df.drop(target_col, axis=1)
y = df[target_col]

selected_subsets = [
    [4, 6, 9, 11],
    [0, 2, 6, 7, 9],
    [2, 4, 10, 12],
    [2, 4, 10, 12],
    [2, 3, 7, 9, 11],
    [6, 7, 9]
]

feature_names = X_full.columns.tolist()

def print_metrics(model, X_test, y_test, model_name, subset_idx):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
    cm = confusion_matrix(y_test, y_pred)

    print(f"Subset {subset_idx + 1} - {model_name} Performance:")
    print(f"  Features: {[feature_names[i] for i in selected_subsets[subset_idx]]}")
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1 Score:       {f1:.4f}")
    print(f"  ROC-AUC Score:  {roc_auc if roc_auc == 'N/A' else f'{roc_auc:.4f}'}")
    print(f"  Confusion Matrix:\n{cm}")
    print("-" * 50)

def main():
    for i, subset_indices in enumerate(selected_subsets):
        X_subset = X_full.iloc[:, subset_indices]
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y, test_size=0.2, random_state=42, stratify=y
        )

        dt = DecisionTreeClassifier(random_state=42)
        knn = KNeighborsClassifier()
        rf = RandomForestClassifier(random_state=42)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        dt.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        print_metrics(dt, X_test, y_test, "Decision Tree", i)
        print_metrics(knn, X_test, y_test, "K-Nearest Neighbors", i)
        print_metrics(rf, X_test, y_test, "Random Forest", i)
        print_metrics(xgb, X_test, y_test, "XGBoost", i)

if __name__ == "__main__":
    main()
