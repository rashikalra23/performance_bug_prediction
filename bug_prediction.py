import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from scipy import stats
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from google.colab import drive
from imblearn.over_sampling import SMOTE


drive.mount('/content/drive')

# Define the dataset path and algorithms
data_dir = '/content/drive/MyDrive/bug_pred_data'
N = 50  # Number of iterations for statistical reliability
proposed_features = ['num_if_inloop', 'num_loop_inif', 'num_nested_loop', 'num_nested_loop_incrit',
                     'synchronization', 'thread', 'io_in_loop', 'database_in_loop', 'collection_in_loop',
                     'io', 'database', 'collection', 'recursive']

# Define algorithms and parameters
algorithms = {
    'Naive Bayes': (ComplementNB, {'alpha': 0.001}),
    'Logistic Regression': (LogisticRegression, {'max_iter': 100, 'class_weight': 'balanced'}),
    'SVM': (LinearSVC, {'max_iter': 1000, 'class_weight': 'balanced'}),
    'Random Forest': (RandomForestClassifier, {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42}),
    'Decision Tree': (DecisionTreeClassifier, {'class_weight': 'balanced', 'random_state': 42}),
    'XGBoost': (XGBClassifier, {'eval_metric': 'logloss'})
}

def get_correlated_features(dataset):
    numeric_data = dataset.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                colname = correlation_matrix.columns[min(i, j)]
                correlated_features.add(colname)
    return correlated_features



def train_test_model(train_x, train_y, test_x, test_y, algorithm, param_dist):
    # **Handle small minority class by adjusting k_neighbors**
    minority_class_count = train_y.value_counts().min()
    smote_k_neighbors = min(5, minority_class_count - 1)  # Ensure k_neighbors < samples in minority class

    if smote_k_neighbors > 0:  # Apply SMOTE only if valid k_neighbors is possible
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        train_x, train_y = smote.fit_resample(train_x, train_y)

    # Scale features
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # Train the model
    clf = algorithm(**param_dist)
    clf.fit(train_x, train_y)

    # Predict probabilities or decision scores
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(test_x)[:, 1]
    else:
        prob_pos = clf.decision_function(test_x)
        probs = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    # Calculate performance metrics
    fpr, tpr, _ = metrics.roc_curve(test_y, probs, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    predictions = clf.predict(test_x)
    mcc_score = metrics.matthews_corrcoef(test_y, predictions)
    f1_score = metrics.f1_score(test_y, predictions)

    return auc_score, mcc_score, f1_score

def out_of_sample_evaluation(dataset, N, algorithm, param_dist):
    correlated_features = get_correlated_features(dataset)
    test_dataset = dataset.drop(columns=list(correlated_features), inplace=False).select_dtypes(include=[np.number])
    X = test_dataset.drop(['label'], axis=1)
    y = test_dataset['label']

    scores = {'test_roc_auc': [], 'test_mcc': [], 'test_f1': []}
    for _ in range(N):
        sample_index = resample(dataset.index)
        train_x, train_y = X.loc[sample_index], y.loc[sample_index]
        test_index = np.setdiff1d(dataset.index, sample_index)
        test_x, test_y = X.loc[test_index], y.loc[test_index]

        if all(test_y == 0) or all(train_y == 0):
            continue

        auc, mcc, f1 = train_test_model(train_x, train_y, test_x, test_y, algorithm, param_dist)
        scores['test_roc_auc'].append(auc)
        scores['test_mcc'].append(mcc)
        scores['test_f1'].append(f1)
    return scores

# Prepare results list to store rows
results_list = []

projects = os.listdir(data_dir)
for project in projects:
    project_path = os.path.join(data_dir, project, 'dataset.csv')
    if os.path.exists(project_path):
        dataset = pd.read_csv(project_path)
        dataset = dataset.drop(columns=['File'], errors='ignore')

        for name, (algo, params) in algorithms.items():
            scores = out_of_sample_evaluation(dataset, N, algo, params)
            if scores['test_roc_auc'] and scores['test_mcc'] and scores['test_f1']:
                row = {
                    'project': project,
                    'auc': np.mean(scores['test_roc_auc']),
                    'mcc': np.mean(scores['test_mcc']),
                    'f1': np.mean(scores['test_f1']),
                    'algorithm_metric': f"{name} with_anti-pattern_metrics"
                }
                results_list.append(row)

            remain_dataset = dataset.drop(columns=proposed_features, errors='ignore')
            scores = out_of_sample_evaluation(remain_dataset, N, algo, params)
            if scores['test_roc_auc'] and scores['test_mcc'] and scores['test_f1']:
                row = {
                    'project': project,
                    'auc': np.mean(scores['test_roc_auc']),
                    'mcc': np.mean(scores['test_mcc']),
                    'f1': np.mean(scores['test_f1']),
                    'algorithm_metric': f"{name} without_anti-pattern_metrics"
                }
                results_list.append(row)

results_df = pd.DataFrame(results_list)
results_df.to_csv('/content/drive/MyDrive/bug_pred_results.csv', index=False)
print("Results saved to /content/drive/MyDrive/bug_pred_results.csv")
