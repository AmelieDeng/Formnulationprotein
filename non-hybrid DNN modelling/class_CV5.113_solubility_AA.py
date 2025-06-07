import math
import os
import numpy as np
import torch.optim as optim
#from torchsummary import summary
import pandas as pd
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import copy
import Constants
import params
from Bio import SeqIO
from Dataset.Dataset import load_dataset
from models.gnn_AA import GCN
import argparse
import torch
import torch.nn.functional as F
import time
from torch_geometric.loader import DataLoader
from preprocessing.utils import pickle_save, pickle_load, save_ckp, load_ckp, class_distribution_counter, \
    draw_architecture, compute_roc, get_sequence_from_pdb,create_seqrecord, get_proteins_from_fasta, generate_bulk_embedding, fasta_to_dictionary
import matplotlib.pyplot as plt
import warnings
import random
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Subset
from tools import summarize_model_with_hierarchy_and_info, scale_form_one, scale_AA, scale_form_one_refit, scale_AA_refit, EarlyStopping_class, plot_training_curves, plot_training_curves_refit, get_params_with_weight_decay

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--data_path', type=str, default="data", help="Path to data files")
parser.add_argument('--output', type=str, default="output", help="File to save output")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def filter_dict(d, key_no):
    for key, value in d.items():
        if  key != key_no:
             print(f"{key}: {value}")


# 数据加载
def create_dataloader(dataset, indices, batch_size, shuffle, num_workers = 4):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, 
        generator=torch.Generator().manual_seed(SEED)
    )

def calculate_metrics(predictions, targets, dataset):
    predictions = F.softmax(torch.tensor(predictions), dim=1).numpy()
    predictions_classes = np.argmax(predictions, axis=1)
    targets = targets.flatten()  

    accuracy = accuracy_score(targets, predictions_classes)
    class_report = classification_report(targets, predictions_classes, output_dict=True)
    class_metrics = {}
    for class_label, metrics in class_report.items():
        if isinstance(metrics, dict): 
            class_metrics[f"{dataset}Class_{class_label}_Precision"] = round(metrics['precision'], 4)
            class_metrics[f"{dataset}Class_{class_label}_Recall"] = round(metrics['recall'], 4)
            class_metrics[f"{dataset}Class_{class_label}_F1"] = round(metrics['f1-score'], 4)

    return {
        f"{dataset}Accuracy": round(accuracy, 4),
        f"{dataset}Precision": round(class_report['macro avg']['precision'], 4),
        f"{dataset}Recall": round(class_report['macro avg']['recall'], 4),
        f"{dataset}F1": round(class_report['macro avg']['f1-score'], 4),
        **class_metrics,  
    }

def train_model(model, optimizer, scheduler, early_stopping, criterion, data_loader, device, epochs, fold, trial, time_log_file, train_curve_plot_path):
    best_model_state = None
    best_result = None
    best_epoch, best_val_loss = 0, 0
    epoch_results = []
    best_result = []
    trial_start_time = time.time()

    train_losses, test_losses, val_losses = [], [], []
    train_accs, test_accs, val_accs = [], [], []

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()
            train_loss = 0
            train_predictions, train_targets = [], []


            for data, target in data_loader['train']:                

                optimizer.zero_grad()
                target = target.long().to(device)
                output = model(data.to(device))


                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_predictions.append(output.detach().cpu().numpy())
                train_targets.append(target.detach().cpu().numpy())
            train_loss /= len(data_loader['train'])
            train_metrics = calculate_metrics(np.concatenate(train_predictions, axis=0), np.concatenate(train_targets, axis=0), "train_")

            val_loss, test_loss = 0, 0
            val_predictions, val_targets = [], []
            test_predictions, test_targets = [], []

            with torch.no_grad():
                model.eval()
                for data, target in data_loader['valid']:

                    target_val = target.long().to(device)
                    output = model(data.to(device))
                    loss = criterion(output, target_val)
                    val_loss += loss.item()
                    val_predictions.append(output.detach().cpu().numpy())
                    val_targets.append(target_val.detach().cpu().numpy())

                for data, target in data_loader['test']:

                    target_test = target.long().to(device)
                    output = model(data.to(device))
                    loss = criterion(output, target_test)
                    test_loss += loss.item()
                    test_predictions.append(output.detach().cpu().numpy())
                    test_targets.append(target_test.detach().cpu().numpy())

            val_loss /= len(data_loader['valid'])
            test_loss /= len(data_loader['test'])
            val_metrics = calculate_metrics(np.concatenate(val_predictions, axis=0), np.concatenate(val_targets, axis=0), "val_")
            test_metrics = calculate_metrics(np.concatenate(test_predictions, axis=0), np.concatenate(test_targets, axis=0), "test_")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accs.append(train_metrics['train_Accuracy'])
            val_accs.append(val_metrics['val_Accuracy'])
            test_accs.append(test_metrics['test_Accuracy'])            

            epoch_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **train_metrics,
                'val_loss': val_loss,
                **val_metrics,
                'test_loss': test_loss,
                **test_metrics
            })

            if val_metrics['val_F1'] > best_val_loss:
                best_val_loss = val_metrics['val_F1']
                best_epoch = epoch
                best_model_state = {
                    'epoch': epoch + 1,
                    'valid_loss_min': val_loss,
                    'state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                }
                best_result = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                **train_metrics,
                'val_loss': val_loss,
                **val_metrics,
                'test_loss': test_loss,
                **test_metrics
                }
                                              
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Fold {fold}, Trial {trial}, Epoch {epoch + 1}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Train Accuracy={train_metrics['train_Accuracy']:.4f}, "
                  f"Train Recall={train_metrics['train_Recall']:.4f}, "
                  f"Train Precision={train_metrics['train_Precision']:.4f}, "
                  f"Train F1={train_metrics['train_F1']:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Accuracy={val_metrics['val_Accuracy']:.4f}, "
                  f"Val Recall={val_metrics['val_Recall']:.4f}, "
                  f"Val Precision={val_metrics['val_Precision']:.4f}, "
                  f"Val F1={val_metrics['val_F1']:.4f}, "
                  f"Test Loss={test_loss:.4f}, "
                  f"Test Accuracy={test_metrics['test_Accuracy']:.4f}, "
                  f"Test Recall={test_metrics['test_Recall']:.4f}, "
                  f"Test Precision={test_metrics['test_Precision']:.4f}, "
                  f"Test F1={test_metrics['test_F1']:.4f}, "
                  f"Time {epoch_time:.2f}")

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics['val_F1'])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")

            if early_stopping(val_metrics['val_F1'], model, epoch):
                break             

    plot_training_curves(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, train_curve_plot_path, epoch)
    trial_end_time = time.time()
    trial_duration = trial_end_time - trial_start_time
    print(f"Trial {trial} in Fold {fold} completed in {trial_duration:.2f} seconds.")

    # Log trial duration
    with open(time_log_file, "a") as f:
        f.write(f"Trial {trial} in Fold {fold} duration: {trial_duration:.2f} seconds\n")

    return best_model_state, best_epoch, epoch_results, best_result


def Five_validation(resultdir, dataset, device, model_class, params,grid_params, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=SEED)
    all_results = []
    best_split_result = None
    total_start_time = time.time()
    time_log_file = f"{resultdir}/csv/nested_cv_time_log.txt"

    split_performance = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        fold_start_time = time.time()
        print(f"\n=== Fold {fold + 1}/{k} ===")
        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)

        train_indices, val_indices = train_test_split(range(len(train_data)), test_size=0.3, random_state=SEED)
        inner_train_data = Subset(train_data, train_indices)
        val_data = Subset(train_data, val_indices)
        inner_train_data, val_data, test_data = scale_form_one(inner_train_data, val_data, test_data)
        inner_train_data, val_data, test_data = scale_AA(inner_train_data, val_data, test_data)
        best_trial_result = None

        for trial_id, (num_epochs, lr, batch_size) in enumerate(
            [(ep, lr, bs) for ep in grid_params['num_epochs'] for lr in grid_params['lr'] for bs in grid_params['batch_size']]):

            train_loader = create_dataloader(inner_train_data, list(range(len(inner_train_data))), batch_size, shuffle=True)
            val_loader = create_dataloader(val_data, list(range(len(val_data))), batch_size, shuffle=False)
            test_loader = create_dataloader(test_data, list(range(len(test_data))), batch_size, shuffle=False)

            print('========================================')
            print(f'# training samples: {len(inner_train_data)}')
            print(f'# val samples: {len(val_data)}')
            print(f'# test samples: {len(test_data)}')
            print(f'# trial_id: {trial_id + 1}, lr: {lr}, batch_size: {batch_size}')
            print('========================================')

            data_loader = {'train': train_loader, 'valid': val_loader, 'test': test_loader}

            model = model_class(**params).to(device)
            param = get_params_with_weight_decay(model, weight_decay = grid_params['wd'])
            optimizer = torch.optim.Adam(param, lr=lr)

            criterion = torch.nn.CrossEntropyLoss()
            save_path = f"{resultdir}/model/fold{fold + 1}_stop_trial{trial_id + 1}.pth"
            early_stopping = EarlyStopping_class(save_path, patience=100)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=0.0001)

            train_curve_plot_path = f"{resultdir}/model/fold{fold + 1}_stop_trial{trial_id + 1}.png"
            best_model_state, best_epoch, results, best_epoch_result = train_model(
                model, optimizer, scheduler, early_stopping, criterion, data_loader, device, num_epochs, fold + 1, trial_id + 1, time_log_file, train_curve_plot_path
            )

            trial_result = {
                'fold': fold + 1,
                'trial': trial_id + 1,
                'num_epochs': num_epochs,
                'lr': lr,
                'batch_size': batch_size,
                'best_epoch': best_epoch + 1,
                #'results': results,
                'state_dict': best_model_state,
                'best_epoch_result': best_epoch_result
                
            }
            epochs_result = results
            # save each trail epoch result
            pd.DataFrame(epochs_result).to_csv(f"{resultdir}/csv/fold{fold + 1}_trial{trial_id + 1}_epochs_summary.csv", index=False)
            
            if best_trial_result is None or trial_result['best_epoch_result']['val_F1'] > best_trial_result['best_epoch_result']['val_F1']:
                best_trial_result = trial_result
                best_trial_epochs_result = epochs_result

            if best_split_result is None or best_trial_result['best_epoch_result']['val_F1'] > best_split_result['best_epoch_result']['val_F1']:
                best_split_result = best_trial_result
      
            #print(f"Trial {trial_id + 1}, Epochs={num_epochs}, LR={lr}, Batch={batch_size}: Best Epoch={best_epoch + 1}, best_trial_epochs_result = {best_trial_epochs_result}")

            model_save_path = f"{resultdir}/trail_best_epoch_model/fold{fold + 1}_trial{trial_id + 1}_best_epoch{best_epoch + 1}.pth"
            torch.save(best_model_state, model_save_path)

        all_results.append(best_trial_result)

        # Save best trial for each fold
        pd.DataFrame(best_trial_epochs_result).to_csv(f"{resultdir}/csv/fold{fold + 1}_best_trial_summary.csv", index=False)
        model_save_path = f"{resultdir}/model/fold{fold + 1}_best_model.pth"
        torch.save(best_trial_result['state_dict'], model_save_path)
        print('best_trial_result:')
        filter_dict(best_trial_result, key_no = 'state_dict')


        split_performance.append(best_trial_result['best_epoch_result'])

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        print(f"Fold {fold + 1} completed in {fold_duration:.2f} seconds.")
        with open(time_log_file, "a") as f:
            f.write(f"Fold {fold + 1} duration: {fold_duration:.2f} seconds\n")

        


    performance_df = pd.DataFrame(split_performance)
    mean_performance = performance_df.mean()
    std_performance = performance_df.std()
    print("\nAverage Performance Across Folds:")
    print(mean_performance)
    print("\nStandard Deviation of Performance Across Folds:")
    print(std_performance)
    performance_df.to_csv(f"{resultdir}/csv/split_performance_summary.csv", index=False)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total Nested CV completed in {total_duration:.2f} seconds.")
    with open(time_log_file, "a") as f:
        f.write(f"Total Nested CV duration: {total_duration:.2f} seconds\n")

    print(f"\n=== Refit Best Split ===")
    print(f"Best Split Fold: {best_split_result['fold']}")
    filter_dict(best_split_result, key_no = 'state_dict')


    # Save overall summary
    for metrics in all_results:
        if 'state_dict' in metrics:
            del metrics['state_dict']
    
    pd.DataFrame(all_results).to_csv(f"{resultdir}/csv/nested_cv_summary.csv", index=False)
    print("\n=== Nested Cross-Validation Complete ===")

    print("All trial and fold summaries are saved to CSV files.")

    # Return final model and its performance metrics
    return

# Example function call
root = "/media/ouyang/backup_disk/output/0data"


mapping = {
    "0_solubility_class_AA": {"csv": "solubility_merged_proteins_20250522_AA.csv", "formcol": -5},
}
num_class = 4
esm_p = "0_solubility_class_AA"
resultdir = f"/media/ouyang/backup_disk/output/0result/class/{esm_p}"

formadd = mapping[esm_p]["csv"]
forminfo = mapping[esm_p]["formcol"]

formdata_p = f"/media/ouyang/backup_disk/output/0data/formdata/{formadd}"
form = pd.read_csv(formdata_p)

labels_df = form.iloc[:, 21: forminfo]
formfeatures = labels_df.shape[1]

AA_df = form.iloc[:, 1: 21]
AAfeatures = AA_df.shape[1]

#18
dataset = load_dataset(root=root, esm_p=esm_p, formdata_p=formdata_p)
print('========================================')
print(f'# proteins: {len(dataset)}')

print('========================================')

model_params = {
    'hidden': 32,
    'input_features_size': 321,
    'num_classes': num_class,
    'edge_type': 'cbrt',
    'edge_features': 1, # edge feature dim
    'egnn_layers': 1,
    'layers': 1,
    'device': 'cuda',
    'formnums': formfeatures,
    'AAnums': AAfeatures,    
    'wd': 1e-8 #5e-4
}


grid_params = {
    'num_epochs': [1000],
    'lr': [0.0012],
    'batch_size': [8, 16, 24, 36, 48, 64],
    'wd': 1e-8
}


five_results = Five_validation(
    resultdir = resultdir,
    dataset=dataset,      
    device=device,        
    model_class=GCN, 
    params=model_params,   
    grid_params=grid_params,
    k=5                    
)

