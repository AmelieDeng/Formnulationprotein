#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score,make_scorer,accuracy_score, recall_score, f1_score
import pandas as pd
from scoring_module import evaluate_subset_class, lgb_feature_plot
from lightgbm import LGBMClassifier
import optuna
import joblib
import os
import seaborn as sns
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

##viscocity
# data = pd.read_csv('.../finaldataset/viscocity_merged_proteins.csv')
# y = data.iloc[:, -6].values
# X = data.iloc[:, :-6].values
# print(data.iloc[:, :-6].head())

# feature_names = data.iloc[:,:-6].columns.tolist()

##collidial
# data = pd.read_csv('.../finaldataset/collidial_merged_proteins.csv')
# y = data.iloc[:, -4].values
# X = data.iloc[:, :-4].values
# print(data.iloc[:, :-4].head())
# feature_names = data.iloc[:,:-4].columns.tolist()

##Tm
data = pd.read_csv('../../finaldataset/conformation_merged_proteins_class.csv')
y = data.iloc[:, -4].values
X = data.iloc[:, :-4].values
print(data.iloc[:, :-4].head())
feature_names = data.iloc[:,:-4].columns.tolist()



results_file_path = '../new_result/all_ml_result/lightgbm_conformation.csv'
try:
    results_df = pd.read_csv(results_file_path)
except FileNotFoundError:

    results_df = pd.DataFrame(columns=[
        'fold', 'train_pre', 'train_acc', 'train_recall', 'train_f1',
        'val_pre', 'val_acc', 'val_recall', 'val_f1',
        'test_pre', 'test_acc', 'test_recall', 'test_f1', 'best_params'
    ])


def objective(trial, X_train, y_train, X_val, y_val, random_state = 42):

    n_estimators = trial.suggest_int('n_estimators',30,100,5)
    # learning_rate = trial.suggest_float('learning_rate',0.1,0.2)
    num_leaves = trial.suggest_int("num_leaves",4,24,2)
    max_depth = trial.suggest_int("max_depth",4,24,2)


    # min_child_weight = trial.suggest_float('min_child_weight',1, 10)
    # subsample = trial.suggest_float('subsample',0.1, 0.9)
    # min_child_samples = trial.suggest_int("min_child_samples",5, 60, 5)   
    # colsample_bytree = trial.suggest_float('colsample_bytree',0.1, 0.9)

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

                    # colsample_bytree = colsample_bytree,
                    # min_child_samples = min_child_samples,
                    # subsample = subsample,
                    # min_child_weight =min_child_weight,

                    random_state=random_state)

    new.fit(X_train, y_train)
    train = evaluate_subset_class(new, X_train, y_train, 'train')
    test = evaluate_subset_class(new, X_val, y_val, 'test')

    return train['accuracy'], test['recall']


def optimizer_optuna(n_trials, algo, X_train, y_train,  X_val, y_val):


    algo = optuna.samplers.TPESampler(n_startup_trials = 40, n_ei_candidates = 24)
    study = optuna.create_study(sampler = algo,directions=["minimize", "maximize"])                                
    study.optimize(lambda trial: objective(trial, X_train, y_train,  X_val, y_val), n_trials=n_trials, show_progress_bar=False, timeout=300)
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


for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):

    print('fold_idx', fold_idx)
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]
    

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)


    best_params = optimizer_optuna(30,X_train, y_train, X_val, y_val)


    model = LGBMClassifier(**best_params, random_state=42)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    pre_val = precision_score(y_val, val_pred)
    acc_val = accuracy_score(y_val, val_pred)
    recall_val = recall_score(y_val, val_pred)
    f1_val = f1_score(y_val, val_pred)
    # report = classification_report(y_val, val_pred, target_names= labels)

    pre_val_scores.append(pre_val)
    acc_val_scores.append(acc_val)
    recall_val_scores.append(recall_val)
    f1_val_scores.append(f1_val)

    train_pre = precision_score(y_train, train_pred)
    train_acc = accuracy_score(y_train, train_pred)
    train_recall = recall_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred)
    # train_report = classification_report(y_train, train_pred, target_names= labels)

    train_pre_scores.append(train_pre)
    train_acc_scores.append(train_acc)
    train_recall_scores.append(train_recall)
    train_f1_scores.append(train_f1)

    pre = precision_score(y_test, test_pred)
    acc = accuracy_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    # test_report = classification_report(y_test, test_pred, target_names= labels)

    pre_scores.append(pre)
    acc_scores.append(acc)
    recall_scores.append(recall)
    f1_scores.append(f1)


    results_df = results_df.append({
    'fold': fold_idx + 1,
    'train_pre': np.round(train_pre, 3), 'train_acc': np.round(train_acc, 3), 'train_recall': np.round(train_recall, 3), 'train_f1': np.round(train_f1, 3),
    'val_pre': np.round(pre_val, 3), 'val_acc': np.round(acc_val, 3), 'val_recall': np.round(recall_val, 3), 'val_f1': np.round(f1_val, 3),
    'test_pre': np.round(pre, 3), 'test_acc': np.round(acc, 3), 'test_recall': np.round(recall, 3), 'test_f1': np.round(f1, 3), 'best_params':best_params
    }, ignore_index=True)

    results_df.to_csv(results_file_path, index=False)

pre_mean = np.mean(pre_scores)
pre_std = np.std(pre_scores)
acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)
recall_mean = np.mean(recall_scores)
recall_std = np.std(recall_scores)
f1_mean = np.mean(f1_scores)
f1_std = np.std(f1_scores)


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

print(f'Mean precision across folds: {pre_mean:.3f} ± {pre_std:.3f}')
print(f'Mean ACC across folds: {acc_mean:.3f} ± {acc_std:.3f}')
print(f'Mean recall across folds: {recall_mean:.3f} ± {recall_std:.3f}')
print(f'Mean f1 across folds: {f1_mean:.3f} ± {f1_std:.3f}')

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
    'test_f1': f'{f1_mean:.3f} ± {f1_std:.3f}',
    'best_params': ""
}, ignore_index=True)
results_df.to_csv(results_file_path, index=False)