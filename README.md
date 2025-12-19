***AI-Based Network Intrusion Detection System (NIDS)***


Python

scikit-learn

Streamlit

Hopsworks

A machine learning-based Network Intrusion Detection System that detects malicious network traffic using supervised and unsupervised algorithms. The project demonstrates how AI can overcome limitations of traditional rule-based IDS by achieving high accuracy, high recall, and low false positive rates.

**🚀 Features**


**Binary classification**: Normal vs Intrusive traffic

**Multiple ML models compared**: Random Forest, SVM, Logistic Regression, Isolation Forest

Model selection based on lowest False Positive Rate

**Full preprocessing pipeline**: encoding, scaling, duplicate removal, class imbalance handling (SMOTE)

**Professional MLOps**: Model registered and versioned in Hopsworks Model Registry

Interactive demo using Streamlit showcasing test results

**Clean visualization**: Confusion matrix, key metrics, highlighted intrusions


**📊 Dataset**

NSL-KDD (improved version of KDD Cup 1999)

Contains 41 features + label

Labeled as normal or various attack types (mapped to binary: normal=0, attack=1)


**🏆 Results (on test set)**


Accuracy: ~98%

Precision (Intrusive): ~97-99%

Recall (Intrusive): ~95-98%

F1-Score: ~97-98%

False Positive Rate: ~1-3% (very low — minimal false alarms)


The best-performing model (typically Random Forest) is saved and deployed via Hopsworks.

**🛠️ Tech Stack**

Python 3.10+

scikit-learn – Model training and evaluation

pandas, numpy – Data processing

imbalanced-learn – SMOTE for class imbalance

Hopsworks – Model registry and MLOps

Streamlit – Interactive results dashboard

Matplotlib/Seaborn – Visualizations


**📁 Project Structure**

textIDS/
├── app.py                          # Streamlit dashboard (main demo)
├── model_training.py               # Training, evaluation, model selection & Hopsworks upload
├── feature_selection.py            # Feature selection and preprocessing pipeline
├── preprocessing.py                # Data cleaning and transformation functions
├── test_data_with_predictions.csv  # Saved test results for demo
├── model/                          # Local model files (gitignored)
└── README.md                       # This file

**🚀 Quick Start**

1. Clone the repository git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection
2. Install dependencies
Bashpip install -r requirements.txt
(Create a requirements.txt with: scikit-learn, pandas, streamlit, hopsworks, imbalanced-learn, matplotlib, seaborn)
3. Run the training script (one time)
Bashpython model_training.py
This will:

Train and compare models
Select the best one (lowest FPR)
Save results to test_data_with_predictions.csv
Upload the model to Hopsworks Model Registry

4. Launch the interactive demo
Bashstreamlit run app.py
The app will:

Load the best model directly from Hopsworks
Display performance metrics
Show confusion matrix
Highlight intrusive predictions

🎯 Why This Project Matters
Traditional signature-based IDS:

Require constant manual rule updates
Ineffective against zero-day attacks
High false positive rates

This AI-based approach:

Automatically learns patterns from data
Adapts to new attack types
Reduces false alarms
Scales with more training data



***👤 Author***
Taha Faisal
December 2025
