import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, url_for, flash
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
SELECTED_INDICES = [4, 6, 9, 11]

MODELS = {
    "Decision Tree": "models/decision_tree_model.h5",
    "KNN": "models/knn_model.h5",
    "Random Forest": "models/random_forest_model.h5",
    "XGBoost": "models/xgboost_model.h5"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_with_models(input_data):
    votes = []
    for path in MODELS.values():
        model = joblib.load(path)
        pred = model.predict(input_data[:, SELECTED_INDICES])[0]
        votes.append(pred)
    final = int(sum(votes) >= 2)
    return votes, final

def evaluate_uploaded_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.drop("target", axis=1)
    y = df["target"]
    all_preds = []
    results = {}

    for name, path in MODELS.items():
        model = joblib.load(path)
        X_sel = X.iloc[:, SELECTED_INDICES]
        y_pred = model.predict(X_sel)
        all_preds.append(y_pred)
        results[name] = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "f1": round(f1_score(y, y_pred), 4),
            "roc": round(roc_auc_score(y, y_pred), 4),
            "cm": confusion_matrix(y, y_pred).tolist()
        }

    combined = (np.sum(all_preds, axis=0) >= 2).astype(int)
    results["Combined"] = {
        "accuracy": round(accuracy_score(y, combined), 4),
        "f1": round(f1_score(y, combined), 4),
        "roc": round(roc_auc_score(y, combined), 4),
        "cm": confusion_matrix(y, combined).tolist()
    }

    return results

# Routes
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def do_login():
    user = request.form['username']
    pwd = request.form['password']
    if user == "Lance" and pwd == "Lance@1234":
        session['user'] = user
        return redirect(url_for("home"))
    else:
        flash("Invalid credentials", "danger")
        return redirect(url_for("login"))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("home.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if 'user' not in session:
        return redirect(url_for("login"))

    result = None
    votes = {}
    combined_result = None

    if request.method == 'POST':
        values = np.array([float(request.form[f]) for f in ALL_FEATURES]).reshape(1, -1)
        model_preds, final = predict_with_models(values)
        votes = dict(zip(MODELS.keys(), model_preds))
        combined_result = "Heart Disease Detected" if final else "No Heart Disease"

    return render_template("manual.html", result=combined_result, votes=votes)

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if 'user' not in session:
        return redirect(url_for("login"))

    results = None
    if request.method == 'POST':
        file = request.files['testfile']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            results = evaluate_uploaded_csv(filepath)
        else:
            flash("Please upload a valid CSV file", "danger")

    return render_template("evaluate.html", results=results)

if __name__ == '__main__':
    app.run(debug=True)
