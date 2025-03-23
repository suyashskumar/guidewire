# **Predicting Kubernetes Issues**  

## **Overview**  
This project develops an AI/ML model to predict **Kubernetes cluster failures**, including **pod crashes, resource bottlenecks, and network issues**. The model analyzes **historical and real-time cluster metrics** to anticipate failures.  

## **Project Structure**  
```
Predicting-Kubernetes-Issues  
│── Predicting_Kubernetes_issues.ipynb  
│── balanced_shuffled_traffic.csv
│── Predicting Kubernetes Issues - Documentation.pdf
│── README.md  
```

## **Dataset**  
- **File Used:** `balanced_shuffled_traffic.csv`  
- **Features:** Network traffic metrics, resource utilization  
- **Target Variable:** `Label` (Indicating failure or no failure)  
- **Data Insights:**  
  - Dataset contains **8,621 records** and **60 features**  
  - Balanced class distribution (**52% non-failure, 48% failure**)  
  - Key correlated features include **packet sizes, flow rates, and connection statistics**  

## **Data Preprocessing**  
- **Missing Values:** Checked and removed if found  
- **Feature Selection:**  
  - Selected features based on correlation with the target variable  
  - Examples: `Bwd Packet Length Std`, `Flow Bytes/s`, `Flow Packets/s`  

## **Model Training**  
- **Algorithm Used:** Random Forest Classifier  
- **Steps Taken:**  
  - Splitting dataset into training and testing sets  
  - Standardizing numerical features  
  - Training with selected features  

## **Errors Encountered**  
- **NameError:** `loaded_model` was not defined before making predictions  
  - **Fix:** Assigned trained model (`rf_classifier`) to `loaded_model`  
- **Correlation Calculation Issue:** Non-numeric values caused failure in `df.corr()`  
  - **Fix:** Computed correlation only for numerical features  
