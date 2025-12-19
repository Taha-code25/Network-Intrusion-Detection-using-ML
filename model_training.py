from feature_selection import feature_selection
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import hopsworks
import joblib
import os
import numpy as np
import pandas as pd

X_train, X_test, y_train, y_test = feature_selection()


models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "IsolationForest": IsolationForest(contamination=0.1, random_state=42)
}


best_name = None
best_model = None
best_fpr = np.inf
best_metrics = {}

print("Training and evaluating models...\n")

for name, model in models.items():
    print(f"Training {name}...")

    if name == "IsolationForest":
        model.fit(X_train) 
    else:
        model.fit(X_train, y_train)
    

    if name == "IsolationForest":
        preds = model.predict(X_test)
        preds = np.where(preds == -1, 1, 0) 
    else:
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | FPR: {fpr:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")

    if fpr < best_fpr:
        best_fpr = fpr
        best_name = name
        best_model = model
        best_metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "false_positive_rate": fpr
        }
    

print(f"Best model: {best_name} with FPR = {best_fpr:.4f}")
preds = best_model.predict(X_test)
df_results = pd.DataFrame({
    'actual_label': ['Normal' if y == 0 else 'Intrusive' for y in y_test],
    'prediction': preds,
    'prediction_label': ['Normal' if p == 0 else 'Intrusive' for p in preds]
})
df_results.to_csv("test_data_with_predictions.csv", index=False)
print("Test results saved for demo!")
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "best_nids_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved locally at: {model_path}")


project = hopsworks.login(api_key_value=os.getenv("API_KEY"))  
mr = project.get_model_registry()

model_obj = mr.sklearn.create_model(
    name="nids_best_model",
    version=None,  
    metrics=best_metrics,
    description=f"Best intrusion detection model: {best_name} (lowest FPR on NSL-KDD test set)",
)

model_obj.save(model_path)
print("Model successfully uploaded to Hopsworks Model Registry! 🎉")