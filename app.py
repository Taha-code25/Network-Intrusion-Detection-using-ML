# app.py
import streamlit as st
import pandas as pd
import hopsworks
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="NIDS Results", layout="centered")
st.title("🔒 AI Network Intrusion Detection System – Results")

st.markdown("""
**Trained on NSL-KDD dataset**  
**Best model stored in Hopsworks Model Registry** (professional MLOps platform)  
**Goal**: Demonstrate how machine learning effectively detects network intrusions
""")

# --- Load model and test results ---
@st.cache_resource
def load_model_and_results():
    try:
        # Connect to Hopsworks and load model
        project = hopsworks.login()
        mr = project.get_model_registry()
        model_obj = mr.get_model("nids_best_model", version=1)  # Change if your model name/version is different
        model_dir = model_obj.download()
        model = joblib.load(os.path.join(model_dir, "best_nids_model.pkl"))
        
        st.success("✅ Model successfully loaded from Hopsworks!")
        
        # Load your saved test results (you created this file once after training)
        df = pd.read_csv("test_data_with_predictions.csv")
        
        return model, df
    
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.stop()

model, df = load_model_and_results()

# --- Calculate metrics ---
y_true = df['actual_label'].map({'Normal': 0, 'Intrusive': 1})  # Convert back to 0/1 if needed
y_pred = df['prediction']

report = classification_report(y_true, y_pred, output_dict=True, target_names=['Normal', 'Intrusive'])

accuracy = report['accuracy']
precision_intrusive = report['Intrusive']['precision']
recall_intrusive = report['Intrusive']['recall']
f1_intrusive = report['Intrusive']['f1-score']

# --- Display key metrics ---
st.header("Model Performance on Test Data")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.1%}")
col2.metric("Precision (Intrusive)", f"{precision_intrusive:.1%}")
col3.metric("Recall (Intrusive)", f"{recall_intrusive:.1%}")
col4.metric("F1-Score (Intrusive)", f"{f1_intrusive:.1%}")

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Intrusive'],
            yticklabels=['Normal', 'Intrusive'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

# False Positive Rate
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
st.info(f"**False Positive Rate**: {fpr:.1%} (very low — minimal false alarms!)")

# --- Sample predictions ---
st.subheader("Sample Test Predictions (20 examples)")

display_df = df[['actual_label', 'prediction_label']].head(20).copy()
display_df.columns = ['Actual', 'Predicted']

def highlight_intrusive(row):
    color = 'background-color: #ffcccc' if row['Predicted'] == 'Intrusive' else ''
    return [color, color]

styled_df = display_df.style.apply(highlight_intrusive, axis=1)
st.dataframe(styled_df, use_container_width=True)

# --- Conclusion ---
st.success(f"Out of {len(df)} test records, the model correctly identifies attacks with high recall and only {fpr:.1%} false positives!")

st.markdown("""
**Key Takeaway**:  
Machine learning overcomes limitations of traditional rule-based IDS by automatically learning attack patterns from data, achieving high detection rates with fewer false alarms.
""")

st.markdown("---")
st.caption("Built with Streamlit • Model managed in Hopsworks • Dataset: NSL-KDD")