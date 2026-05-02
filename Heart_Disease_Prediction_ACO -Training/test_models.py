import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

ALL_FEATURES = [
    ("age", "Age"),
    ("sex", "Sex (1 = male, 0 = female)"),
    ("cp", "Chest Pain Type (0–3)"),
    ("trestbps", "Resting Blood Pressure (mm Hg)"),
    ("chol", "Serum Cholesterol (mg/dl)"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl (1=true)"),
    ("restecg", "Resting ECG Results (0–2)"),
    ("thalach", "Max Heart Rate Achieved"),
    ("exang", "Exercise Induced Angina (1 = yes; 0 = no)"),
    ("oldpeak", "ST Depression induced by Exercise"),
    ("slope", "Slope of the ST Segment"),
    ("ca", "Major Vessels Colored by Fluoroscopy (0–3)"),
    ("thal", "Thalassemia (1 = normal; 2 = fixed; 3 = reversible)")
]

SELECTED_INDICES = [4, 6, 9, 11]

MODELS = {
    "Decision Tree": "models/decision_tree_model.pkl",
    "KNN": "models/knn_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

def get_manual_input():
    print("\nEnter patient data:")
    inputs = []
    for short, full in ALL_FEATURES:
        val = float(input(f"{full}: "))
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)

def load_and_predict(model_path, input_data):
    model = joblib.load(model_path)
    return model.predict(input_data[:, SELECTED_INDICES])[0]

def evaluate_on_test_data():
    df = pd.read_csv("data/heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Accuracy Report on Test Set (using selected features) ---")

    all_preds = []

    for model_name, path in MODELS.items():
        model = joblib.load(path)
        X_test_selected = X_test.iloc[:, SELECTED_INDICES]
        y_pred = model.predict(X_test_selected.values)
        all_preds.append(y_pred)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n📊 {model_name}")
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Score     : {f1:.4f}")
        print(f"ROC-AUC      : {roc:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    all_preds = np.array(all_preds)
    combined_preds = (np.sum(all_preds, axis=0) >= 2).astype(int)

    acc = accuracy_score(y_test, combined_preds)
    f1 = f1_score(y_test, combined_preds)
    roc = roc_auc_score(y_test, combined_preds)
    cm = confusion_matrix(y_test, combined_preds)

    print(f"\n🧠 Combined Prediction")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC-AUC      : {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

def main():
    print("Choose input mode:")
    print("1. Manual Input")
    print("2. Evaluate on Test Set")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        input_data = get_manual_input()
        print("\n--- Predictions ---")

        votes = []
        for model_name, path in MODELS.items():
            prediction = load_and_predict(path, input_data)
            print(f"{model_name}: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
            votes.append(prediction)

        combined_prediction = int(sum(votes) >= 2)
        print(f"\n🧠 Combined Prediction : {'Heart Disease' if combined_prediction == 1 else 'No Heart Disease'}")

    elif choice == "2":
        evaluate_on_test_data()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
