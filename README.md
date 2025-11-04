# Bank-Term-Deposit-Prediction-ML
A supervised machine learning project that predicts whether a bank customer will subscribe to a term deposit. The project uses data preprocessing, feature encoding, data balancing (SMOTE), and two classification algorithms Random Forest and Support Vector Machine with hyperparameter tuning to improve accuracy and precision.

# Bank Term Deposit Prediction using Random Forest and SVM

This project applies **Supervised Machine Learning** techniques to predict whether a bank customer will subscribe to a term deposit.  
It is based on the **UCI Bank Marketing Dataset** and explores data preprocessing, feature transformation, scaling, data balancing using SMOTE, and model performance improvement using hyperparameter tuning.

---

## 📊 Project Overview
- **Goal:** Predict if a client subscribes to a term deposit (`y` variable).
- **Techniques Used:** Classification (Random Forest, SVM)
- **Dataset Size:** 4521 rows × 17 columns (after encoding: 49 columns)
- **Environment:** Google Colab / Jupyter Notebook

---
---
### 🏷️ Keywords
machine-learning, supervised-learning, random-forest, svm, classification, python, scikit-learn, smote, data-science

## ⚙️ Steps and Methods

1. **Data Preparation**
   - Imported and explored the dataset using Pandas and NumPy.
   - Handled categorical features using `map()` and `get_dummies()`.
   - Performed scaling using `StandardScaler()`.

2. **Data Splitting**
   - Split dataset into training (70%) and testing (30%) using `train_test_split()`.

3. **Data Balancing**
   - Addressed class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).

4. **Model Building**
   - Built and trained **Random Forest Classifier** and **Support Vector Machine**.
   - Evaluated using Accuracy, Precision, Recall, and F1 Score.

5. **Hyperparameter Tuning**
   - Used **GridSearchCV** to find optimal `n_estimators`, `max_features` for Random Forest, and `kernel`, `C` for SVM.

6. **Model Evaluation**
   - Random Forest: Accuracy = **89.46%**
   - SVM: Accuracy = **87.39%**
   - Evaluation metrics: Confusion Matrix, Precision, Recall, F1 Score.

---

## 🧮 Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `imblearn`
- `matplotlib` (optional for visualizations)

---

## 🧠 Key Learnings
- How to preprocess categorical and numerical data.
- How to handle imbalanced datasets using SMOTE.
- The importance of hyperparameter tuning.
- Comparative analysis between ensemble (RF) and kernel-based (SVM) models.

---

## 📈 Results Summary
| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|---------|-----------|
| Random Forest | 89.46% | 55.05% | 32.23% | 40.66% |
| SVM | 87.39% | 43.79% | 44.07% | 43.93% |

---

## 📑 Files Included
- **`Bank_Prediction_Model.py`** — Python code implementation  
- **`Report_Rahul_Pagar.pdf`** — Detailed report of the methodology and results  

---

## Author
**Rahul Pagar**  
Masters in Business Analytics

🔗 [LinkedIn Profile](https://www.linkedin.com/in/rahul-pagar1993)

---

## 🏁 Conclusion
Both models show promising accuracy, but Random Forest achieved higher overall performance and is recommended for real-world deployment due to its robustness and ensemble nature.
