import torch
import numpy as np
import pandas as pd
import os
import pickle
from torch_geometric.loader import DataLoader
from models.gnn_refit import GCN
from Dataset.Dataset import load_dataset


def scale_form_one_prediction(scaler_path, test_data):

    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].form_one
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].form_one = torch.tensor(scaled_feature, dtype=torch.float)
        return dataset

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
            
    test_data = transform_dataset(test_data, scaler)

    return test_data
def create_dataloader(dataset, batch_size, shuffle, num_workers=4, seed=42):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # 优化传输到 GPU 的效率
        generator=torch.Generator().manual_seed(seed)
    )

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def extract_embeddings_and_predictions(model, dataloader, device, save_path):
    model.eval()
    model.to(device)

    all_embeddings = []
    all_predictions = []  # 存储预测结果
    all_targets = []  # 存储目标变量

    with torch.no_grad():  # 关闭梯度计算，节省显存
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)  # 送入 GPU/CPU
            predictions, final_embedding = model(data)  # 获取预测结果和 embedding
            final_embedding = torch.cat(final_embedding, dim=1)

            all_embeddings.append(final_embedding.cpu())  # 转移到 CPU
            all_predictions.append(predictions.cpu())  # 存储预测结果                      
            all_targets.append(target.cpu())  # 存储目标变量

    # 转换为 NumPy 格式

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()  # (样本数, 特征维度)
    #all_predictions = torch.cat(all_predictions, dim=0).numpy()  # (样本数, 1)
    all_targets = torch.cat(all_targets, dim=0).numpy()  # (样本数,)

    embeddings_df = pd.DataFrame(all_embeddings)  # embedding 数据
    #predictions_df = pd.DataFrame(all_predictions, columns=["Prediction"])  # 预测数据
    targets_df = pd.DataFrame(all_targets, columns=["Target"])  # 真实标签
#all    
    embeddings_df.to_csv(os.path.join(save_path, "allembeddings.csv"), index=False)
    #predictions_df.to_csv(os.path.join(save_path, "allpredictions.csv"), index=False)
    targets_df.to_csv(os.path.join(save_path, "alltargets.csv"), index=False)
#test
    #embeddings_df.to_csv(os.path.join(save_path, "embeddings.csv"), index=False)
   # predictions_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)
    #targets_df.to_csv(os.path.join(save_path, "targets.csv"), index=False)
    print(f"Embedding 和预测结果提取完成，已保存到 {save_path}")

def load_saved_data(save_path):
    X = np.load(os.path.join(save_path, "embeddings.npy"))  # 读取 embedding
    preds = np.load(os.path.join(save_path, "predictions.npy"))  # 读取预测结果
    y_path = os.path.join(save_path, "targets.npy")
    y = np.load(y_path) if os.path.exists(y_path) else None  # 读取标签（如果存在）

    print("Embedding 维度:", X.shape)  # (样本数, 特征维度)
    print("预测结果维度:", preds.shape)  # (样本数, 1)
    if y is not None:
        print("目标变量数量:", y.shape)
    return X, preds, y

# 示例调用
if __name__ == "__main__":

    mapping = {
        "0_conformation": {"csv": "conformation_merged_proteins_class.csv", "formcol": -3},
        "0_solubility": {"csv": "solubility_merged_proteins_20250425.csv", "formcol": -4},
        "0_collodial": {"csv": "collidial_merged_proteins.csv", "formcol": -3},
        "0_viscosity": {"csv": "viscocity_merged_proteins.csv", "formcol": -5},
        "0_solubility_class": {"csv": "solubility_merged_proteins_20250522.csv", "formcol": -5},        
    }

    esm_p = "0_solubility_class"
    formadd = mapping[esm_p]["csv"]
    forminfo = mapping[esm_p]["formcol"]

    formdata_p = f"/media/ouyang/backup_disk/output/0data/formdata/{formadd}"
    form = pd.read_csv(formdata_p)
    labels_df = form.iloc[:, 1: forminfo]
    formfeatures = labels_df.shape[1]

    model_params = {
        'hidden': 32,
        'input_features_size': 321,
        'num_classes': 4,
        'edge_type': 'cbrt',
        'edge_features': 1, # edge feature dim
        'egnn_layers': 1,
        'layers': 1,
        'device': 'cuda',
        'formnums': formfeatures,
        'wd': 1e-8 #5e-4
    }
    batch_size = 4
    shuffle = False
    num_workers = 4
    seed = 42

    testmapping = {
        "exp_kd":{"csv": "experiment/kd_2025118.csv", "formcol": -2},
        "exp_tm":{"csv": "experiment/Tm_2025118_class.csv", "formcol": -2},
        "exp_viscosity":{"csv": "experiment/viscosity_2025118.csv", "formcol": -3},
        "exp_solubility":{"csv": "experiment/solubility_20250425.csv", "formcol": -3},     
        "exp_solubility_class":{"csv": "experiment/solubility_20250522.csv", "formcol": -4},             
    }   
    
    testesm_p = "exp_solubility_class"
    testroot = "/media/ouyang/backup_disk/output/0data"
    testresultdir = f"/media/ouyang/backup_disk/output/0result/class/{testesm_p}"
    
   
    scaler_path = f"/media/ouyang/backup_disk/output/0result/class/{testesm_p}/scaler.pkl"        
    
    model_path = f"/media/ouyang/backup_disk/output/0result/class/{testesm_p}/trail_best_epoch_model/_trial6_best_epoch244.pth"   
    state_dict = torch.load(model_path)

    
    model = GCN(**model_params)
    
    model.load_state_dict(state_dict['state_dict'])
    
#test    
    testformadd = testmapping[testesm_p]["csv"]
    testformdata_p = f"/media/ouyang/backup_disk/output/0data/formdata/{testformadd}"

    #testdataset = load_dataset(root=testroot, esm_p=testesm_p, formdata_p=testformdata_p)    
    #testdata = scale_form_one_prediction(scaler_path, testdataset)

#all    
    dataset = load_dataset(root=testroot, esm_p=esm_p, formdata_p=formdata_p)    
    testdata = scale_form_one_prediction(scaler_path, dataset)    
#    
    dataloader = create_dataloader(testdata, batch_size, shuffle, num_workers, seed)
    
    save_path = f"{testresultdir}/saved_results"  # 指定保存路径
    os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    extract_embeddings_and_predictions(model, dataloader, device, save_path)
    
    #X, preds, y = load_saved_data(save_path)
    
#change model_path, save_csv_name, task_name, exp_name
    
    
    
    
