#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sklearn.metrics as sm
from sklearn.metrics import precision_score,make_scorer,accuracy_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

def ten_CV(model, X_train, y_train, random_state=1996):
    cv = KFold(n_splits=10, shuffle=True, random_state=1996)
    scoring = {'mae': 'neg_mean_absolute_error',
               'mse': 'neg_mean_squared_error',
               'rmse': 'neg_root_mean_squared_error',
               'r2': 'r2'}

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs = -1)
    # print(cv_results['test_r2'])

    mae_mean = -cv_results['test_mae'].mean()
    mae_std = cv_results['test_mae'].std()
    mse_mean = -cv_results['test_mse'].mean()
    mse_std = cv_results['test_mse'].std()
    rmse_mean = -cv_results['test_rmse'].mean()
    rmse_std = cv_results['test_rmse'].std()
    r2_mean = cv_results['test_r2'].mean()
    r2_std = cv_results['test_r2'].std()
    # r2 = cv_results['test_r2']

    MAE =  "{:.3f} ± {:.3f}".format(mae_mean, mae_std)
    MSE = "{:.3f} ± {:.3f}".format(mse_mean, mse_std)
    RMSE = "{:.3f} ± {:.3f}".format(rmse_mean, rmse_std)
    R2 = "{:.3f} ± {:.3f}".format(r2_mean,r2_std)
    # print("r2", r2)
    # print("10CV_MAE", MAE, "10CV_MSE", MSE, "10CV_RMSE", RMSE, "10CV_R2", R2)

    ten_cv_result = {"10CV_MAE":MAE, "10CV_MSE": MSE, "10CV_RMSE": RMSE, "10CV_R2": R2}
    # ten_cv_result = {"10CV_MAE":MAE, "10CV_MSE": MSE, "10CV_RMSE": RMSE, "10CV_R2": R2, "R2_mean": r2_mean, "MSE_mean": mse_mean}

    return ten_cv_result

def evaluate_subset(model, X, y_true, subsetname = 'subsetname'):

    y_pred = model.predict(X)

    def subset_score(y_true, y_pred, subsetname = 'subsetname'):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred,squared=False)

        # print("{}_MAE: {:.3f}".format(subsetname, mae), "{}_MSE: {:.3f}".format(subsetname, mse), "{}_RMSE: {:.3f}".format(subsetname, rmse), "{}_R2: {:.3f}".format(subsetname, r2))

        results_dict = {
                        "dataset_name": subsetname,
                        "MAE": np.around(mae, 3),
                        "MSE": np.around(mse, 3),
                        "RMSE": np.around(rmse, 3),
                        "R^2": np.around(r2, 3)
                        }
        return results_dict

    scores = subset_score(y_true, y_pred, subsetname)
  
    return scores




def evaluate_subset_class(model, X, y_true, subsetname = 'subsetname'):

    y_pred = model.predict(X)

    def subset_score_class(y_true, y_pred, subsetname = 'subsetname'):
        pre = precision_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # print("{}_MAE: {:.3f}".format(subsetname, mae), "{}_MSE: {:.3f}".format(subsetname, mse), "{}_RMSE: {:.3f}".format(subsetname, rmse), "{}_R2: {:.3f}".format(subsetname, r2))

        results_dict = {
                        "dataset_name": subsetname,
                        "precision": np.around(pre, 3),
                        "accuracy": np.around(acc, 3),
                        "recall": np.around(recall, 3),
                        "f1": np.around(f1, 3)
                        }
        return results_dict

    scores = subset_score_class(y_true, y_pred, subsetname)
    # a = sm.confusion_matrix(y_true, y_pred)
    # print(a)
    return scores



def ten_CV_class(model, X_train, y_train, random_state=1996):
    cv = KFold(n_splits=10, shuffle=True, random_state=1996)
    scoring = {'precision': 'precision',
               'accuracy': 'accuracy',
               'recall': 'recall',
               'f1': 'f1'}

    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

    precision_mean = cv_results['test_precision'].mean()
    precision_std = cv_results['test_precision'].std()
    accuracy_mean = cv_results['test_accuracy'].mean()
    accuracy_std = cv_results['test_accuracy'].std()
    recall_mean = cv_results['test_recall'].mean()
    recall_std = cv_results['test_recall'].std()
    f1_mean = cv_results['test_f1'].mean()
    f1_std = cv_results['test_f1'].std()

    precision =  "{:.3f} ± {:.3f}".format(precision_mean, precision_std)
    accuracy = "{:.3f} ± {:.3f}".format(accuracy_mean, accuracy_std)
    recall = "{:.3f} ± {:.3f}".format(recall_mean, recall_std)
    f1 = "{:.3f} ± {:.3f}".format(f1_mean, f1_std)

    ten_cv_result = {"10CV_Precision": precision, "10CV_Accuracy": accuracy, "10CV_Recall": recall, "10CV_F1": f1, "recall_mean":recall_mean, "precision_mean":precision_mean}

    return ten_cv_result


def print_result(model, X, y, X_train, y_train, X_test, y_test, random_state):
    print(evaluate_subset(model, X_train, y_train, 'train'))
    print(evaluate_subset(model, X_test, y_test, 'test'))
    print(ten_CV(model, X, y, random_state))
    

def print_result_class(model, X_train, y_train, X_test, y_test, random_state):
    print(evaluate_subset_class(model, X_train, y_train, 'train'))
    print(ten_CV_class(model, X_train, y_train, random_state))
    print(evaluate_subset_class(model, X_test, y_test, 'test'))




def lgb_feature_plot(model, save_dir):
    lgb.plot_importance(model, max_num_features=10, importance_type='gain') 
    plt.savefig(save_dir,dpi=300, bbox_inches="tight")
    plt.title("Feature importance (%)")
    # plt.show()

def scatter_plot(model, X_train, y_train, X_test, y_test, min, max, save_dir):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    plt.figure(figsize=(7,3.5))
    plt.rcParams['font.family'] = ['Times New Roman']# 设置字体
    plt.rcParams['font.size'] = 12# 设置字体大小

    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(y_train, pred_train,s = 12,label="Training set")
    ax1.plot([min,max],[min,max],"k--")
    ax1.set_ylim(bottom=min,top=max)
    ax1.set_xlim(left=min, right=max)
    # ax1.set_yticks(np.arange(30,110,20))
    # ax1.set_xticks(np.arange(30,110,20)) 

    plt.xlabel('Experimental values (%)')
    plt.ylabel('Predicted values (%)')
    plt.title("Training set", fontsize=16, y=1.1)

    ax2 = plt.subplot(1, 2, 2)
    plt.scatter(y_test, pred_test,s = 12,label="Test set")
    ax2.plot([min,max],[min,max],"k--")
    ax2.set_ylim(bottom=min,top=max)
    ax2.set_xlim(left=min, right=max)
    # ax2.set_yticks(np.arange(30,110,20))
    # ax2.set_xticks(np.arange(30,110,20)) 

    ax2.set_xlabel('Experimental values (%)')
    ax2.set_ylabel('Predicted values (%)')
    plt.title("Test set", fontsize=16, y=1.1)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()

    plt.savefig(save_dir, dpi=300, bbox_inches='tight')