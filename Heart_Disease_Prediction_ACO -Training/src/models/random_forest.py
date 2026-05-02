import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SELECTED_FEATURES = [4, 6, 9, 11]

def train_and_save_model(data_path="data/heart.csv", model_path="models/random_forest_model.h5"):
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    X_selected = X.iloc[:, SELECTED_FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    y_pred = model.predict(X_test)
    print(f"[Random Forest] Accuracy: {accuracy_score(y_test, y_pred):.4f}")

if __name__ == "__main__":
    train_and_save_model()
