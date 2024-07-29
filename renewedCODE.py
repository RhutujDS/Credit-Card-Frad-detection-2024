# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 14:14:52 2024

@author: rhutu
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
from sklearn.impute import SimpleImputer
import shap

file_path = r'C:/Users/rhutu/OneDrive/Desktop/projectData/german.data.csv'

# Loading the dataset
data = pd.read_csv('C:/Users/rhutu/OneDrive/Desktop/projectData/german.data.csv', delimiter=' ', header=None)


# Displaying basic information about the dataset
print(data.info())
print(data.head())

# Checking for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Identifying categorical and numerical columns by their data types
categorical_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
numerical_cols = [1, 4, 7, 10, 12, 15, 17]

# Debugging: Printing columns to verify
print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

# Ensuring all indices are within range
assert all(col < data.shape[1] for col in categorical_cols)
assert all(col < data.shape[1] for col in numerical_cols)

# Data Distribution Plot
plt.figure(figsize=(18, 12))  # Adjusting the size for better clarity
data[numerical_cols].hist(bins=30, edgecolor='k', alpha=0.7, layout=(3, 3), figsize=(18, 12))  # Specifying the layout
plt.suptitle('Numerical Feature Distributions', fontsize=16)
plt.show()


# Data Imbalance Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=data.iloc[:, -1])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Correlation Heatmap
# Extracting numerical data
numerical_data = data[numerical_cols]

# Creating the correlation matrix
corr_matrix = numerical_data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap')
plt.show()

# Summarizing strong correlations
# Considering correlations above 0.8 or below -0.8 as strong
threshold = 0.8
strong_corrs = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]
strong_corrs = strong_corrs[strong_corrs < 1].stack().reset_index()
strong_corrs.columns = ['Variable 1', 'Variable 2', 'Correlation']
strong_corrs = strong_corrs.sort_values(by='Correlation', ascending=False)

# Displaying the strong correlations
print("Strong Correlations (>= 0.8 or <= -0.8):")
print(strong_corrs)
# Creating preprocessing pipelines for both numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundling preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Separating features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting and transforming the data using the preprocessing pipeline
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Initializing the models
logreg = LogisticRegression(max_iter=1000)
logreg_l1 = LogisticRegressionCV(Cs=10, penalty='l1', solver='saga', max_iter=1000, cv=5, random_state=42)
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True)

# Training the models
logreg.fit(X_train_preprocessed, y_train)
logreg_l1.fit(X_train_preprocessed, y_train)
rf.fit(X_train_preprocessed, y_train)
svm.fit(X_train_preprocessed, y_train)

# Predicting on the test set
logreg_preds = logreg.predict(X_test_preprocessed)
logreg_l1_preds = logreg_l1.predict(X_test_preprocessed)
rf_preds = rf.predict(X_test_preprocessed)
svm_preds = svm.predict(X_test_preprocessed)

# Evaluating the models
print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_preds))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("Logistic Regression with L1 Regularization Accuracy:", accuracy_score(y_test, logreg_l1_preds))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logreg_preds))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds))

# F1 Score
print("Logistic Regression F1 Score:", f1_score(y_test, logreg_preds, average='binary', pos_label=2))
print("Logistic Regression with L1 Regularization F1 Score:", f1_score(y_test, logreg_l1_preds, average='binary', pos_label=2))
print("Random Forest F1 Score:", f1_score(y_test, rf_preds, average='binary', pos_label=2))
print("SVM F1 Score:", f1_score(y_test, svm_preds, average='binary', pos_label=2))

# ROC AUC
logreg_probs = logreg.predict_proba(X_test_preprocessed)[:, 1]
logreg_l1_probs = logreg_l1.predict_proba(X_test_preprocessed)[:,1]
rf_probs = rf.predict_proba(X_test_preprocessed)[:, 1]
svm_probs = svm.predict_proba(X_test_preprocessed)[:, 1]

print("Logistic Regression ROC AUC:", roc_auc_score(y_test, logreg_probs))
print("Logistic Regression with L1 Regularization ROC AUC:", roc_auc_score(y_test, logreg_l1_probs))
print("Random Forest ROC AUC:", roc_auc_score(y_test, rf_probs))
print("SVM ROC AUC:", roc_auc_score(y_test, svm_probs))

# ROC Curve
logreg_fpr, logreg_tpr, _ = roc_curve(y_test, logreg_probs, pos_label=2)
logreg_l1_fpr, logreg_l1_tpr, _ = roc_curve(y_test, logreg_l1_probs, pos_label=2)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs, pos_label=2)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs, pos_label=2)

plt.figure(figsize=(10, 6))
plt.plot(logreg_fpr, logreg_tpr, label='Logistic Regression (area = %0.2f)' % roc_auc_score(y_test, logreg_probs))
plt.plot(logreg_l1_fpr, logreg_l1_tpr, label='Logistic Regression L1 (area = %0.2f)' % roc_auc_score(y_test, logreg_l1_probs))
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % roc_auc_score(y_test, rf_probs))
plt.plot(svm_fpr, svm_tpr, label='SVM (area = %0.2f)' % roc_auc_score(y_test, svm_probs))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
logreg_cm = confusion_matrix(y_test, logreg_preds)
rf_cm = confusion_matrix(y_test, rf_preds)
svm_cm = confusion_matrix(y_test, svm_preds)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(logreg_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 3, 3)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()


########################################


# Hyperparameter tuning with GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

# Initializing RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initializing GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2, error_score='raise')

# Fitting GridSearchCV
try:
    grid_search_rf.fit(X_train_preprocessed, y_train)
    best_rf = grid_search_rf.best_estimator_

    # Best parameters and score
    print("Best parameters for Random Forest:", grid_search_rf.best_params_)
    print("Best score for Random Forest:", grid_search_rf.best_score_)

    # Evaluate on test set
    best_rf_preds = best_rf.predict(X_test_preprocessed)
    print("Best Random Forest Test Accuracy:", accuracy_score(y_test, best_rf_preds))
    print("\nBest Random Forest Classification Report:\n", classification_report(y_test, best_rf_preds))
except ValueError as e:
    print(f"Error during hyperparameter tuning: {e}")

#############################################################

# Assuming the RandomForestClassifier has been trained as best_rf
# Training the Random Forest model if not already done
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train_preprocessed, y_train)

# Initializing SHAP explainer for Random Forest
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test_preprocessed)

# Summary plot
shap.summary_plot(shap_values, X_test_preprocessed, plot_type="bar")
##################################################


# Gradient boosting and XGBoost

# Creating preprocessing pipelines for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Separating features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Preprocessing the features
X_preprocessed = preprocessor.fit_transform(X)

# Debugging: Printing shapes of preprocessed arrays
print("Shape of X_preprocessed:", X_preprocessed.shape)

# Mapping labels [1, 2] to [0, 1]
y = y.map({1: 0, 2: 1})

# Adding polynomial features after preprocessing
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_preprocessed)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initializing the models
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, eval_metric='logloss')

# Training the models
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predicting on the test set
gb_preds = gb.predict(X_test)
xgb_preds = xgb.predict(X_test)

# Evaluating the models
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

print("\nGradient Boosting Classification Report:\n", classification_report(y_test, gb_preds))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# F1 Score
print("Gradient Boosting F1 Score:", f1_score(y_test, gb_preds, average='binary', pos_label=1))
print("XGBoost F1 Score:", f1_score(y_test, xgb_preds, average='binary', pos_label=1))

# ROC AUC
gb_probs = gb.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]

print("Gradient Boosting ROC AUC:", roc_auc_score(y_test, gb_probs))
print("XGBoost ROC AUC:", roc_auc_score(y_test, xgb_probs))

# ROC Curve
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs, pos_label=1)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs, pos_label=1)

plt.figure(figsize=(10, 6))
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting (area = %0.2f)' % roc_auc_score(y_test, gb_probs))
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost (area = %0.2f)' % roc_auc_score(y_test, xgb_probs))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
gb_cm = confusion_matrix(y_test, gb_preds)
xgb_cm = confusion_matrix(y_test, xgb_preds)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Generating feature names
num_features = [f"num_{i}" for i in range(len(numerical_cols))]
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
all_features = np.concatenate([num_features, cat_features])
poly_features = poly.get_feature_names_out(all_features)

# Defining the number of top features to display
top_n = 20  # Number of top features to display

# Feature Importance for Gradient Boosting
gb_importances = gb.feature_importances_
indices = np.argsort(gb_importances)[::-1][:top_n]

plt.figure(figsize=(15, 8))
plt.title("Top 20 Feature Importances - Gradient Boosting")
plt.bar(range(top_n), gb_importances[indices], align="center")
plt.xticks(range(top_n), [poly_features[i] for i in indices], rotation=90)
plt.xlim([-1, top_n])
plt.show()

# Feature Importance for XGBoost
xgb_importances = xgb.feature_importances_
indices = np.argsort(xgb_importances)[::-1][:top_n]

plt.figure(figsize=(15, 8))
plt.title("Top 20 Feature Importances - XGBoost")
plt.bar(range(top_n), xgb_importances[indices], align="center")
plt.xticks(range(top_n), [poly_features[i] for i in indices], rotation=90)
plt.xlim([-1, top_n])
plt.show()

# SHAP for Gradient Boosting
explainer_gb = shap.Explainer(gb)
shap_values_gb = explainer_gb(X_test)

# Summary plot for Gradient Boosting
shap.summary_plot(shap_values_gb, X_test, plot_type="bar")

# SHAP for XGBoost
explainer_xgb = shap.Explainer(xgb)
shap_values_xgb = explainer_xgb(X_test)

# Summary plot for XGBoost
shap.summary_plot(shap_values_xgb, X_test, plot_type="bar")

################################################################

# Hyperparameter tuning for XGBoost and Gradient Boosting

# Creating preprocessing pipelines for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
    ])

# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Preprocessing the features
X_preprocessed = preprocessor.fit_transform(X)

# Mapping labels [1, 2] to [0, 1]
y = y.map({1: 0, 2: 1})

# Adding polynomial features after preprocessing
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_preprocessed)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0]
}

gb_grid_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=42),
                                    param_distributions=gb_param_grid,
                                    scoring='accuracy',
                                    cv=3,
                                    n_jobs=-1,
                                    n_iter=20,
                                    random_state=42)

gb_grid_search.fit(X_train, y_train)
best_gb = gb_grid_search.best_estimator_

# Hyperparameter tuning for XGBoost
xgb_param_dist = {
    'n_estimators': randint(100, 200),
    'learning_rate': uniform(0.01, 0.09),  # Ensure learning_rate values are within [0.01, 0.1)
    'max_depth': randint(3, 5),
    'min_child_weight': randint(1, 3),
    'subsample': uniform(0.8, 0.2),  # Ensure subsample values are within [0.8, 1.0)
    'colsample_bytree': uniform(0.8, 0.2)  # Ensure colsample_bytree values are within [0.8, 1.0)
}

xgb_random_search = RandomizedSearchCV(estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
                                       param_distributions=xgb_param_dist,
                                       scoring='accuracy',
                                       cv

=3,
                                       n_jobs=-1,
                                       n_iter=20,
                                       random_state=42)

xgb_random_search.fit(X_train, y_train)
best_xgb = xgb_random_search.best_estimator_

# Predicting on the test set
gb_preds = best_gb.predict(X_test)
xgb_preds = best_xgb.predict(X_test)

# Evaluating the models
print("Gradient Boosting Accuracy:", accuracy_score(y_test, gb_preds))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))

print("\nGradient Boosting Classification Report:\n", classification_report(y_test, gb_preds))
print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# F1 Score
print("Gradient Boosting F1 Score:", f1_score(y_test, gb_preds, average='binary', pos_label=1))
print("XGBoost F1 Score:", f1_score(y_test, xgb_preds, average='binary', pos_label=1))

# ROC AUC
gb_probs = best_gb.predict_proba(X_test)[:, 1]
xgb_probs = best_xgb.predict_proba(X_test)[:, 1]

print("Gradient Boosting ROC AUC:", roc_auc_score(y_test, gb_probs))
print("XGBoost ROC AUC:", roc_auc_score(y_test, xgb_probs))

# ROC Curve
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs, pos_label=1)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs, pos_label=1)

plt.figure(figsize=(10, 6))
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting (area = %0.2f)' % roc_auc_score(y_test, gb_probs))
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost (area = %0.2f)' % roc_auc_score(y_test, xgb_probs))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
gb_cm = confusion_matrix(y_test, gb_preds)
xgb_cm = confusion_matrix(y_test, xgb_preds)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Generating feature names
num_features = [f"num_{i}" for i in range(len(numerical_cols))]
cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out()
all_features = np.concatenate([num_features, cat_features])
poly_features = poly.get_feature_names_out(all_features)

# Defining the number of top features to display
top_n = 20  # Number of top features to display

# Feature Importance for Gradient Boosting
gb_importances = best_gb.feature_importances_
indices = np.argsort(gb_importances)[::-1][:top_n]

plt.figure(figsize=(15, 8))
plt.title("Top 20 Feature Importances - Gradient Boosting")
plt.bar(range(top_n), gb_importances[indices], align="center")
plt.xticks(range(top_n), [poly_features[i] for i in indices], rotation=90)
plt.xlim([-1, top_n])
plt.show()

# Feature Importance for XGBoost
xgb_importances = best_xgb.feature_importances_
indices = np.argsort(xgb_importances)[::-1][:top_n]

plt.figure(figsize=(15, 8))
plt.title("Top 20 Feature Importances - XGBoost")
plt.bar(range(top_n), xgb_importances[indices], align="center")
plt.xticks(range(top_n), [poly_features[i] for i in indices], rotation=90)
plt.xlim([-1, top_n])
plt.show()

# SHAP for Gradient Boosting
explainer_gb = shap.Explainer(best_gb)
shap_values_gb = explainer_gb(X_test)

# SHAP for XGBoost
explainer_xgb = shap.Explainer(best_xgb)
shap_values_xgb = explainer_xgb(X_test)



#Detailed breakdown for each feature for each sample, you can use:
shap_values_gb_array = shap_values_gb.values
shap_values_xgb_array = shap_values_xgb.values

# To print or analyze specific SHAP values:
# For instance, print SHAP values for the first instance in the test set
print("SHAP values for the first instance - Gradient Boosting:")
print(shap_values_gb_array[0])

print("SHAP values for the first instance - XGBoost:")
print(shap_values_xgb_array[0])
##########################################
shap.summary_plot(shap_values_gb, X_test)
shap.summary_plot(shap_values_xgb, X_test)