from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import feature_extraction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

### Modelling ###
print("---Logistic Regression Model--")
log_model = LogisticRegression(C=0.01, penalty='l2', solver='lbfgs')
log_model.fit(feature_extraction.X_train_final, feature_extraction.y_train)
print(f"Training score: {log_model.score(feature_extraction.X_train_final, feature_extraction.y_train)}")
print(f"Test score: {log_model.score(feature_extraction.X_test_final, feature_extraction.y_test)}")

log_model_pred = log_model.predict(feature_extraction.X_test_final)
log_model_proba = log_model.predict_proba(feature_extraction.X_test_final)

print("---Evaluate Logistic Regression Model--")
print(f"Accuracy: {accuracy_score(feature_extraction.y_test, log_model_pred)}")
print(f"Precision: {precision_score(feature_extraction.y_test, log_model_pred, average='weighted')}")
print(f"Recall: {recall_score(feature_extraction.y_test, log_model_pred, average='weighted')}")
print(f"F1 score: {f1_score(feature_extraction.y_test, log_model_pred, average='weighted')}")
print(f"ROC-AUC score: {roc_auc_score(feature_extraction.y_test, log_model_proba, average='weighted', multi_class='ovr')}")


print("---Random Forest Classifier Model--")
rfc_model = RandomForestClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)
rfc_model.fit(feature_extraction.X_train_final, feature_extraction.y_train)
print(f"Training score: {rfc_model.score(feature_extraction.X_train_final, feature_extraction.y_train)}")
print(f"Test score: {rfc_model.score(feature_extraction.X_test_final, feature_extraction.y_test)}")

rfc_model_pred = rfc_model.predict(feature_extraction.X_test_final)
rfc_model_proba = log_model.predict_proba(feature_extraction.X_test_final)

print("---Evaluate Random Forest Classification Model--")
print(f"Accuracy: {accuracy_score(feature_extraction.y_test, rfc_model_pred)}")
print(f"Precision: {precision_score(feature_extraction.y_test, rfc_model_pred, average='weighted')}")
print(f"Recall: {recall_score(feature_extraction.y_test, rfc_model_pred, average='weighted')}")
print(f"F1 score: {f1_score(feature_extraction.y_test, rfc_model_pred, average='weighted')}")
print(f"ROC-AUC score: {roc_auc_score(feature_extraction.y_test, rfc_model_proba, average='weighted', multi_class='ovr')}")


