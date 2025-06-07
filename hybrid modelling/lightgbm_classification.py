#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score,make_scorer,accuracy_score, recall_score, f1_score, classification_report
import pandas as pd
from scoring_module import evaluate_subset_class, lgb_feature_plot, variance_threshold_selection, correlation_threshold_selection,mutual_information_selection
from lightgbm import LGBMClassifier
import optuna
import joblib
import os
import seaborn as sns
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


embeddata = pd.read_csv('./allembeddings.csv')
X_embed = embeddata.iloc[:, :]
ndarray_df = pd.DataFrame(X_embed)

data = pd.read_csv('./solubility_merged_proteins_20250522.csv')
y = data.iloc[:, -6].values
# X = data.iloc[:, :-6].values

 
combined_df = pd.concat([ndarray_df,data.iloc[:, :20]], axis=1)
print(combined_df.shape)
X_raw, _ = correlation_threshold_selection(combined_df)
X = pd.concat([X_raw, data.iloc[:, 20:-6]], axis=1)
X.to_csv('./combined.csv', index = False)


Xdata = pd.read_csv('./combined.csv')
X = Xdata.values

results_file_path = './solubility_ensemble_lightgbm_result.csv'
try:
    results_df = pd.read_csv(results_file_path)
except FileNotFoundError:
 
    results_df = pd.DataFrame(columns=[
        'fold', 'train_pre', 'train_acc', 'train_recall', 'train_f1',
        'val_pre', 'val_acc', 'val_recall', 'val_f1',
        'test_pre', 'test_acc', 'test_recall', 'test_f1', 'best_params'
    ])


def objective(trial, X_train, y_train, X_val, y_val,random_state = 42):

    n_estimators = trial.suggest_int('n_estimators',20,150,5)
    # learning_rate = trial.suggest_float('learning_rate',0.1,0.4)
    num_leaves = trial.suggest_int("num_leaves",4,30,2)
    max_depth = trial.suggest_int("max_depth",4,30,2)


    # min_child_weight = trial.suggest_float('min_child_weight',1, 10)
    # subsample = trial.suggest_float('subsample',0.1, 0.9)

    min_child_samples = trial.suggest_int("min_child_samples",5, 60, 5)   
    colsample_bytree = trial.suggest_float('colsample_bytree',0.1, 0.8)

    reg_alpha = trial.suggest_float('reg_alpha',0.1, 0.9)
    reg_lambda = trial.suggest_float('reg_lambda',0.1, 0.9)

    new = LGBMClassifier(
                    # **params,       
                    n_estimators =  n_estimators,
                    # learning_rate = learning_rate,
                    num_leaves = num_leaves,
                    max_depth = max_depth,

                    reg_lambda = reg_lambda,
                    reg_alpha = reg_alpha,

                    colsample_bytree = colsample_bytree,
                    min_child_samples = min_child_samples,
                    # subsample = subsample,
                    # min_child_weight =min_child_weight,
                    class_weight = 'balanced',

                    random_state=random_state)

    new.fit(X_train, y_train)
    train = evaluate_subset_class(new, X_train, y_train, 'train')
    val = evaluate_subset_class(new, X_val, y_val, 'val')

    return train['recall'], val['f1']


def optimizer_optuna(n_trials, algo, X_train, y_train, X_val, y_val):

 
    algo = optuna.samplers.TPESampler(n_startup_trials = 40, n_ei_candidates = 24)
    study = optuna.create_study(sampler = algo,directions=["minimize", "maximize"])                                
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=False, timeout=300)
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")
    trial_with_highest = min(study.best_trials, key=lambda t: t.values[0])
    print(f"\tvalues: {trial_with_highest.values}")
    best = trial_with_highest_accuracy.params
    params = best
    return params


kf = KFold(n_splits=5, shuffle=True, random_state=42)

pre_scores = []
acc_scores = []
recall_scores = []
f1_scores = []

pre_val_scores = []
acc_val_scores = []
recall_val_scores = []
f1_val_scores = []

train_pre_scores = []
train_acc_scores = []
train_recall_scores = []
train_f1_scores = []

labels = ['class 0', 'class 1', 'class 2', 'class 3']

train_class_metrics = {label: {'precision': [], 'accuracy': [],'recall': [], 'f1-score': []} for label in labels}
val_class_metrics = {label: {'precision': [], 'accuracy': [],'recall': [], 'f1-score': []} for label in labels}
test_class_metrics = {label: {'precision': [],'accuracy': [], 'recall': [], 'f1-score': []} for label in labels}


for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):

    print('fold_idx', fold_idx)
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]
    

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    best_params = optimizer_optuna(30,X_train, y_train, X_val, y_val)



    model = LGBMClassifier(**best_params,class_weight = 'balanced', random_state=42)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    pre_val = precision_score(y_val, val_pred, average='macro')
    acc_val = accuracy_score(y_val, val_pred)
    recall_val = recall_score(y_val, val_pred, average='macro')
    f1_val = f1_score(y_val, val_pred, average='macro')
    report = classification_report(y_val, val_pred, target_names= labels)

    pre_val_scores.append(pre_val)
    acc_val_scores.append(acc_val)
    recall_val_scores.append(recall_val)
    f1_val_scores.append(f1_val)

    train_pred = model.predict(X_train)

    train_pre = precision_score(y_train, train_pred, average='macro')
    train_acc = accuracy_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred, average='macro')
    train_f1 = f1_score(y_train, train_pred, average='macro')
    train_report = classification_report(y_train, train_pred, target_names= labels)

    train_pre_scores.append(train_pre)
    train_acc_scores.append(train_acc)
    train_recall_scores.append(train_recall)
    train_f1_scores.append(train_f1)

    test_pred = model.predict(X_test)
 
    pre = precision_score(y_test, test_pred, average='macro')
    acc = accuracy_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred, average='macro')
    f1 = f1_score(y_test, test_pred, average='macro')
    test_report = classification_report(y_test, test_pred, target_names= labels)
 
    pre_scores.append(pre)
    acc_scores.append(acc)
    recall_scores.append(recall)
    f1_scores.append(f1)
 
    
    results_df = results_df.append({
    'fold': fold_idx + 1,
    'train_pre': train_pre, 'train_acc': train_acc, 'train_recall': train_recall, 'train_f1': train_f1,
    'val_pre': pre_val, 'val_acc': acc_val, 'val_recall': recall_val, 'val_f1': f1_val,
    'test_pre': pre, 'test_acc': acc, 'test_recall': recall, 'test_f1': f1,'best_params':best_params
    }, ignore_index=True)


    train_report = classification_report(y_train, train_pred, target_names=labels, output_dict=True)
    i = 0
    for label in labels:
        
        train_class_metrics[label]['precision'].append(train_report[label]['precision'])
        train_class_metrics[label]['recall'].append(train_report[label]['recall'])
        train_class_metrics[label]['f1-score'].append(train_report[label]['f1-score'])
        # 添加准确率
        denominator = np.sum(y_train == i)
        numerator = np.sum((y_train == i) & (train_pred == i))
        train_class_metrics[label]['accuracy'].append((numerator/denominator))
        i = i+1
       

    val_report = classification_report(y_val, val_pred, target_names=labels, output_dict=True)
    p = 0
    for label in labels:
        val_class_metrics[label]['precision'].append(val_report[label]['precision'])
        val_class_metrics[label]['recall'].append(val_report[label]['recall'])
        val_class_metrics[label]['f1-score'].append(val_report[label]['f1-score'])
        # 添加准确率
        denominator = np.sum(y_val == p)
        numerator = np.sum((y_val == p) & (val_pred == p))
        val_class_metrics[label]['accuracy'].append((numerator/denominator))
        p = p+1

    test_report = classification_report(y_test, test_pred, target_names=labels, output_dict=True)
    z = 0
    for label in labels:
        test_class_metrics[label]['precision'].append(test_report[label]['precision'])
        test_class_metrics[label]['recall'].append(test_report[label]['recall'])
        test_class_metrics[label]['f1-score'].append(test_report[label]['f1-score'])
 
        denominator = np.sum(y_test == z)
        numerator = np.sum((y_test == z) & (test_pred == z))
        test_class_metrics[label]['accuracy'].append((numerator/denominator))
        z = z+1
  

 
pre_mean = np.mean(pre_scores)
pre_std = np.std(pre_scores)
acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)
recall_mean = np.mean(recall_scores)
recall_std = np.std(recall_scores)
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)

 
print(f'Mean precision across folds: {pre_mean:.3f} ± {pre_std:.3f}')
print(f'Mean ACC across folds: {acc_mean:.3f} ± {acc_std:.3f}')
print(f'Mean recall across folds: {recall_mean:.3f} ± {recall_std:.3f}')
print(f'Mean f1 across folds: {f1_mean:.3f} ± {f1_std:.3f}')

train_pre_scores_mean = np.mean(train_pre_scores)
train_pre_scores_std = np.std(train_pre_scores)
train_acc_scores_mean = np.mean(train_acc_scores)
train_acc_scores_std = np.std(train_acc_scores)
train_recall_scores_mean = np.mean(train_recall_scores)
train_recall_scores_std = np.std(train_recall_scores)
train_f1_scores_mean = np.mean(train_f1_scores)
train_f1_scores_std = np.std(train_f1_scores)

print(f'train precision across folds: {train_pre_scores_mean:.3f} ± {train_pre_scores_std:.3f}')
print(f'train ACC across folds: {train_acc_scores_mean:.3f} ± {train_acc_scores_std:.3f}')
print(f'train recall across folds: {train_recall_scores_mean:.3f} ± {train_recall_scores_std:.3f}')
print(f'train f1 across folds: {train_f1_scores_mean:.3f} ± {train_f1_scores_std:.3f}')


pre_val_scores_mean = np.mean(pre_val_scores)
pre_val_scores_std = np.std(pre_val_scores)
acc_val_scores_mean = np.mean(acc_val_scores)
acc_val_scores_std = np.std(acc_val_scores)
recall_val_scores_mean = np.mean(recall_val_scores)
recall_val_scores_std = np.std(recall_val_scores)
f1_val_scores_mean = np.mean(f1_val_scores)
f1_val_scores_std = np.std(f1_val_scores)

print(f'val precision across folds: {pre_val_scores_mean:.3f} ± {pre_val_scores_std:.3f}')
print(f'val ACC across folds: {acc_val_scores_mean:.3f} ± {acc_val_scores_std:.3f}')
print(f'val recall across folds: {recall_val_scores_mean:.3f} ± {recall_val_scores_std:.3f}')
print(f'val f1 across folds: {f1_val_scores_mean:.3f} ± {f1_val_scores_std:.3f}')

results_df = results_df.append({
    'fold': "mean ± std",
    'train_pre': f'{train_pre_scores_mean:.3f} ± {train_pre_scores_std:.3f}', 
    'train_acc': f'{train_acc_scores_mean:.3f} ± {train_acc_scores_std:.3f}', 
    'train_recall': f'{train_recall_scores_mean:.3f} ± {train_recall_scores_std:.3f}', 
    'train_f1': f'{train_f1_scores_mean:.3f} ± {train_f1_scores_std:.3f}',
    'val_pre': f'{pre_val_scores_mean:.3f} ± {pre_val_scores_std:.3f}', 
    'val_acc': f'{acc_val_scores_mean:.3f} ± {acc_val_scores_std:.3f}', 
    'val_recall': f'{recall_val_scores_mean:.3f} ± {recall_val_scores_std:.3f}', 
    'val_f1': f'{f1_val_scores_mean:.3f} ± {f1_val_scores_std:.3f}',
    'test_pre': f'{pre_mean:.3f} ± {pre_std:.3f}', 
    'test_acc': f'{acc_mean:.3f} ± {acc_std:.3f}', 
    'test_recall': f'{recall_mean:.3f} ± {recall_std:.3f}', 
    'test_f1': f'{f1_mean:.3f} ± {f1_std:.3f}'
}, ignore_index=True)

for label in labels:
    train_mean_precision = np.mean(train_class_metrics[label]['precision'])
    train_std_precision = np.std(train_class_metrics[label]['precision'])
    train_mean_recall = np.mean(train_class_metrics[label]['recall'])
    train_std_recall = np.std(train_class_metrics[label]['recall'])
    train_mean_f1 = np.mean(train_class_metrics[label]['f1-score'])
    train_std_f1 = np.std(train_class_metrics[label]['f1-score'])
    train_mean_acc = np.mean(train_class_metrics[label]['accuracy'])
    train_std_acc = np.std(train_class_metrics[label]['accuracy'])

    val_mean_precision = np.mean(val_class_metrics[label]['precision'])
    val_std_precision = np.std(val_class_metrics[label]['precision'])
    val_mean_recall = np.mean(val_class_metrics[label]['recall'])
    val_std_recall = np.std(val_class_metrics[label]['recall'])
    val_mean_f1 = np.mean(val_class_metrics[label]['f1-score'])
    val_std_f1 = np.std(val_class_metrics[label]['f1-score'])
    val_mean_acc = np.mean(val_class_metrics[label]['accuracy'])
    val_std_acc = np.std(val_class_metrics[label]['accuracy'])

    test_mean_precision = np.mean(test_class_metrics[label]['precision'])
    test_std_precision = np.std(test_class_metrics[label]['precision'])
    test_mean_recall = np.mean(test_class_metrics[label]['recall'])
    test_std_recall = np.std(test_class_metrics[label]['recall'])
    test_mean_f1 = np.mean(test_class_metrics[label]['f1-score'])
    test_std_f1 = np.std(test_class_metrics[label]['f1-score'])
    test_mean_acc = np.mean(test_class_metrics[label]['accuracy'])
    test_std_acc = np.std(test_class_metrics[label]['accuracy'])
 
    results_df = results_df.append({
        'fold': f'{label} train mean ± std',
        'train_pre': f"{train_mean_precision:.3f} ± {train_std_precision:.3f}",
        'train_acc': f"{train_mean_acc:.3f} ± {train_std_acc:.3f}",
        'train_recall': f"{train_mean_recall:.3f} ± {train_std_recall:.3f}",
        'train_f1': f"{train_mean_f1:.3f} ± {train_std_f1:.3f}",
        'val_pre': f"{val_mean_precision:.3f} ± {val_std_precision:.3f}",
        'val_acc': f"{val_mean_acc:.3f} ± {val_std_acc:.3f}",
        'val_recall': f"{val_mean_recall:.3f} ± {val_std_recall:.3f}",
        'val_f1': f"{val_mean_f1:.3f} ± {val_std_f1:.3f}",
        'test_pre': f"{test_mean_precision:.3f} ± {test_std_precision:.3f}",
        'test_acc': f"{test_mean_acc:.3f} ± {test_std_acc:.3f}",
        'test_recall': f"{test_mean_recall:.3f} ± {test_std_recall:.3f}",
        'test_f1': f"{test_mean_f1:.3f} ± {test_std_f1:.3f}"
    }, ignore_index=True)
results_df.to_csv(results_file_path, index=False, mode='w')
