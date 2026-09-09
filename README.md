# Customer Churn Prediction Classifier

A complete machine learning project demonstrating binary classification using **Logistic Regression** and **Decision Trees** with comprehensive evaluation metrics, confusion matrix dashboards, and precision-recall analysis.

---

## 📋 Project Overview

**Objective:** Develop a classifier to predict customer churn using demographic and behavioral data.

**Dataset:** Synthetic telecom customer dataset with 1000 samples, 8 features.

**Models Trained:**
1. Logistic Regression
2. Decision Tree Classifier

**Key Deliverables:**
- ✅ 4-panel Confusion Matrix Dashboard
- ✅ Precision-Recall Curves (both models)
- ✅ ROC Curves with AUC scores
- ✅ Feature Importance Analysis
- ✅ Model Comparison Report
- ✅ Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended)**

1. Open Google Colab: https://colab.research.google.com/
2. Upload `churn_classification.ipynb` or click "File → Open notebook"
3. Click "Upload" tab and select the notebook
4. Run all cells (Ctrl+Alt+Enter)
5. All visualizations will appear inline

### **Option 2: Local Environment**

#### **Requirements**
- Python 3.8+
- pip or conda

#### **Installation**

```bash
# Clone or download project files
cd customer-churn-classifier

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### **Running the Notebook**

```bash
# Start Jupyter
jupyter notebook churn_classification.ipynb

# Or use JupyterLab
jupyter lab churn_classification.ipynb
```

#### **Running as Python Script**

```bash
python churn_classification.py
```

---

## 📁 Project Structure

```
customer-churn-classifier/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── churn_classification.ipynb          # Main Jupyter notebook (Colab-ready)
├── churn_classification.py             # Standalone Python script
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv         # Original dataset (generated in notebook)
│   └── processed/
│       └── churn_processed.csv        # Preprocessed data
│
├── models/
│   ├── logistic_regression.pkl        # Trained LR model
│   └── decision_tree.pkl              # Trained DT model
│
├── visualizations/
│   ├── confusion_matrix_dashboard.png  # 4-panel confusion matrix
│   ├── precision_recall_curves.png     # PR curve comparison
│   ├── roc_curves.png                  # ROC curve comparison
│   └── model_comparison.png            # Metrics & feature importance
│
└── results/
    ├── churn_classification_report.txt # Comprehensive metrics summary
    └── interpretation_guide.txt        # How to interpret results
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation & analysis |
| numpy | >=1.21.0 | Numerical operations |
| scikit-learn | >=1.0.0 | ML models & metrics |
| matplotlib | >=3.4.0 | Data visualization |
| seaborn | >=0.11.0 | Statistical visualization |

### **Install All**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🎯 Key Concepts Covered

### **1. Classification**
- Binary classification (Churn: Yes/No)
- Training/testing split (80/20)
- Data scaling & preprocessing

### **2. Logistic Regression**
- Probabilistic classifier
- Outputs probability (0 to 1)
- Decision threshold at 0.5
- Coefficients show feature influence

### **3. Decision Tree**
- Non-linear classifier
- Tree-based decision rules
- Interpretable splits
- Feature importance

### **4. Confusion Matrix**
```
                 Predicted NO    Predicted YES
Actual NO        TN              FP (Type I Error)
Actual YES       FN (Type II)    TP
```
- TP: Correctly identified churners
- TN: Correctly identified non-churners
- FP: False alarm (wasted retention offer)
- FN: Missed churner (lost customer)

### **5. Precision vs Recall**

**Precision** = TP / (TP + FP)
- "Of those we predicted would churn, how many actually did?"
- Use when: False positives are costly (retention budget limits)

**Recall** = TP / (TP + FN)
- "Of those who actually churned, how many did we catch?"
- Use when: False negatives are costly (losing customers)

**Tradeoff:** Increasing one usually decreases the other. Precision-Recall curve visualizes this.

### **6. Precision-Recall Curve**
- Shows Precision vs Recall at different decision thresholds
- Curve closer to top-right = better model
- Used to select optimal threshold based on business needs

### **7. ROC-AUC**
- ROC = Receiver Operating Characteristic
- AUC = Area Under the Curve
- Measures ranking ability: probability of ranking random churner > non-churner
- Range: 0.5 (random) to 1.0 (perfect)

---

## 📊 Model Evaluation Metrics

| Metric | Formula | Interpretation | Range |
|--------|---------|-----------------|-------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | 0-1 |
| **Precision** | TP/(TP+FP) | Positive prediction accuracy | 0-1 |
| **Recall** | TP/(TP+FN) | Actual positive capture rate | 0-1 |
| **F1-Score** | 2(Precision×Recall)/(Precision+Recall) | Harmonic mean | 0-1 |
| **ROC-AUC** | Area under ROC curve | Ranking ability | 0.5-1 |

**General Interpretation:**
- 0.90-1.0: Excellent
- 0.80-0.90: Good
- 0.70-0.80: Fair
- 0.60-0.70: Poor
- <0.60: Very Poor

---

## 🔍 Detailed Workflow

### **Step 1: Data Loading**
```python
# Generates synthetic customer churn dataset
# Features: Age, Tenure, Monthly Charge, Internet Service, Contract Type, etc.
# Target: Churn (0=No, 1=Yes)
```

### **Step 2: Data Preprocessing**
```python
# - Encode categorical variables (Internet_Service, Contract_Type, etc.)
# - Split data: 80% training, 20% testing (stratified)
# - Scale numerical features (StandardScaler)
```

### **Step 3: Model Training**
```python
# Logistic Regression
lr_model.fit(X_train_scaled, y_train)

# Decision Tree
dt_model.fit(X_train_scaled, y_train)
```

### **Step 4: Predictions**
```python
# Class predictions (0 or 1)
y_pred_lr = lr_model.predict(X_test_scaled)

# Probability predictions (0 to 1)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
```

### **Step 5: Evaluation**
```python
# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_proba_lr)
```

### **Step 6: Visualization**
```python
# 1. Confusion Matrix Dashboard (4 panels)
# 2. Precision-Recall Curves
# 3. ROC Curves
# 4. Model Comparison & Feature Importance
```

---

## 📈 Interpreting Results

### **Confusion Matrix Dashboard**

**4-Panel View:**
1. **LR Counts**: Absolute numbers for Logistic Regression
2. **LR Normalized**: Percentages (easier to compare)
3. **DT Counts**: Absolute numbers for Decision Tree
4. **DT Normalized**: Percentages

**What to look for:**
- Diagonal (top-left & bottom-right): Correct predictions ✓
- Off-diagonal: Misclassifications ✗
- High values on diagonal = better model

### **Precision-Recall Curves**

**Interpretation:**
- Curve higher & to the right = better
- Sharp drop = precision/recall tradeoff point
- Adjust decision threshold based on business priority:
  - **Need high recall?** Lower threshold (catch more churners)
  - **Need high precision?** Raise threshold (fewer false alarms)

### **ROC Curves**

**Interpretation:**
- AUC closer to 1.0 = better discriminator
- AUC = 0.5 = random classifier
- Curve above diagonal = better than random
- Larger area under curve = better model

### **Feature Importance (Decision Tree)**

Shows which features most influence churn prediction:
- Higher bar = stronger influence on churn
- Top 3 features are your focus points for retention strategy

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

✅ Binary classification fundamentals  
✅ Train-test split and data preprocessing  
✅ Logistic Regression as a classifier  
✅ Decision Trees for classification  
✅ Confusion matrix interpretation  
✅ Precision & Recall tradeoffs  
✅ Precision-Recall curves  
✅ ROC curves and AUC  
✅ Model comparison and selection  
✅ Business interpretation of ML results  

---

## 💡 Real-World Applications

This project teaches skills applicable to:
- **Telecom**: Customer churn prediction
- **E-commerce**: Customer attrition
- **Finance**: Credit default prediction
- **Healthcare**: Patient readmission risk
- **SaaS**: Subscription cancellation

---

## 🔧 Customization Guide

### **Change the Dataset**

Replace the synthetic data generation with your own CSV:
```python
# Load your data
df = pd.read_csv('your_data.csv')

# Ensure you have:
# - Feature columns (X)
# - Binary target column named 'Churn' (0 or 1)
```

### **Adjust Decision Threshold**

Default is 0.5. Change for business needs:
```python
# More aggressive (catch more churners)
y_pred_custom = (y_pred_proba_lr > 0.4).astype(int)

# More conservative (fewer false alarms)
y_pred_custom = (y_pred_proba_lr > 0.6).astype(int)
```

### **Tune Decision Tree**

```python
# Shallower tree (prevent overfitting)
dt_model = DecisionTreeClassifier(max_depth=3)

# Deeper tree (capture more patterns)
dt_model = DecisionTreeClassifier(max_depth=10)

# Minimum samples to split
dt_model = DecisionTreeClassifier(min_samples_split=20)
```

### **Add More Models**

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Add to your pipeline:
rf_model = RandomForestClassifier(n_estimators=100)
gb_model = GradientBoostingClassifier()

rf_model.fit(X_train_scaled, y_train)
gb_model.fit(X_train_scaled, y_train)
```

---

## 🐛 Troubleshooting

### **ImportError: No module named 'sklearn'**
```bash
pip install scikit-learn
```

### **ModuleNotFoundError: No module named 'seaborn'**
```bash
pip install seaborn
```

### **matplotlib not displaying in Jupyter**
```python
# Add this at the top of notebook
%matplotlib inline
```

### **Out of memory with large dataset**
```python
# Use stratified smaller sample
sample_size = 5000
df_sample = df.sample(n=sample_size, random_state=42)
```

### **Imbalanced dataset warning**
```python
# Use stratified split (already in notebook)
train_test_split(..., stratify=y)

# Or adjust class weights
LogisticRegression(class_weight='balanced')
DecisionTreeClassifier(class_weight='balanced')
```

---

## 📚 Further Reading

- **Confusion Matrix**: https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
- **Precision-Recall**: https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall
- **ROC Curves**: https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
- **Decision Trees**: https://scikit-learn.org/stable/modules/tree.html
- **Logistic Regression**: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

---

## 📝 Output Files Generated

After running the notebook, you'll have:

```
confusion_matrix_dashboard.png       ← 4-panel confusion matrix visualization
precision_recall_curves.png          ← PR curve comparison
roc_curves.png                       ← ROC curve comparison
model_comparison.png                 ← Metrics & feature importance
churn_classification_report.txt      ← Detailed metrics summary
interpretation_guide.txt             ← How to read your results
```

---

## ✨ Key Takeaways

| Concept | Key Insight |
|---------|------------|
| **Confusion Matrix** | Reveals exactly what model gets right/wrong |
| **Precision** | When false positives are costly (retention budget limits) |
| **Recall** | When false negatives are costly (losing customers) |
| **Trade-off** | Can't maximize both—must choose based on business |
| **PR Curve** | Visualizes precision-recall tradeoff at all thresholds |
| **ROC-AUC** | Model's ranking ability (does it rank churners higher?) |
| **Feature Importance** | Shows which features drive churn predictions |

---

## 📞 Questions & Support

For questions about:
- **Setup**: Refer to "Quick Start" section
- **Concepts**: See "Key Concepts Covered" section
- **Interpretation**: See "Interpreting Results" section
- **Errors**: Check "Troubleshooting" section

---

## 📄 License

This project is for educational purposes.

---

**Happy Learning! 🚀**

---

*Last Updated: August 2026*
*Author: Data Science Internship Program*
