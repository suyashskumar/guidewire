# contains the code to train the model and evaluate it

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Drop non-numerical columns
X_train_numeric = X_train.select_dtypes(include=['number'])
X_val_numeric = X_val.select_dtypes(include=['number'])

# Instantiate a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model using the training data
rf_classifier.fit(X_train_numeric, y_train)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_val_numeric)

# Evaluate the model's performance on the validation set
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
# Prepare the testing set by selecting only numerical features
X_test_numeric = X_test.select_dtypes(include=['number'])

# Use the trained rf_classifier to make predictions on the prepared X_test set.
y_pred_test = rf_classifier.predict(X_test_numeric)

# Calculate and print the evaluation metrics for the model's performance on the testing set.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f"Testing Set Accuracy: {accuracy_test:.4f}")
print(f"Testing Set Precision: {precision_test:.4f}")
print(f"Testing Set Recall: {recall_test:.4f}")
print(f"Testing Set F1-score: {f1_test:.4f}")

# Compare the testing set performance with the validation set performance.
# If there is a significant drop in performance on the testing set, it might indicate overfitting.
print("\nComparison of Validation and Testing Set Performance:")
print(f"Accuracy (Validation): {accuracy:.4f}, Accuracy (Testing): {accuracy_test:.4f}")
print(f"Precision (Validation): {precision:.4f}, Precision (Testing): {precision_test:.4f}")
print(f"Recall (Validation): {recall:.4f}, Recall (Testing): {recall_test:.4f}")
print(f"F1-score (Validation): {f1:.4f}, F1-score (Testing): {f1_test:.4f}")

# Based on your evaluation, decide whether the current model is satisfactory for the task or if further model optimization is required.
# (This will be done manually based on the performance comparison.)
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Assuming you have a DataFrame 'df' with your data

# Create and train an Isolation Forest model
model = IsolationForest(contamination='auto', random_state=42)
# 'contamination' can be set to a specific value or 'auto' to estimate it
model.fit(df.select_dtypes(include=['number']))  # Fit to numerical features

# Get anomaly scores
df['anomaly_score'] = model.decision_function(df.select_dtypes(include=['number']))

# Set a threshold for identifying anomalies (you may need to adjust this)
threshold = 0  # Example threshold, adjust as needed

# Create a scatterplot
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.scatter(df.index, df['anomaly_score'], c='blue', label='Normal Data')
plt.scatter(df[df['anomaly_score'] < threshold].index,  # Use '<' for anomalies
            df[df['anomaly_score'] < threshold]['anomaly_score'],  # Use '<' for anomalies
            c='red', label='Anomalies')
plt.xlabel('Data Point Index')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Detection Scatterplot')
plt.legend()
plt.show()
