import torch
import torch.nn as nn 
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def summarize_model_with_hierarchy_and_info(model, data, device="cuda", output_file="./model_summary.txt"):
    model = model.to(device)
    executed_layers = []  # 存储实际执行的层信息
    hooks = []
    output_lines = []  # 保存输出的所有行

    # 注册 hook 捕获实际执行的层信息
    def register_hook(module):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
            elif isinstance(output, (tuple, list)):
                output_shape = [list(o.shape) if isinstance(o, torch.Tensor) else str(o) for o in output]
            else:
                output_shape = str(output)
            num_params = sum(p.numel() for p in module.parameters() if hasattr(p, 'numel'))
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            executed_layers.append({
                "module": module,
                "layer_name": module.__class__.__name__,
                "output_shape": output_shape,
                "num_params": num_params,
                "trainable": "Yes" if trainable > 0 else "No",
            })

        if not isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    data = data.to(device)

    # 执行前向传播
    with torch.no_grad():
        model(data)

    for hook in hooks:
        hook.remove()

    # 递归构建层次结构
    def build_hierarchy(module, depth=0):
        hierarchy = []
        for name, child in module.named_children():
            executed_info = next((info for info in executed_layers if info["module"] is child), None)
            node = {
                "name": name,
                "layer_type": child.__class__.__name__,
                "depth": depth,
                "executed": bool(executed_info),
                "info": executed_info,
                "children": build_hierarchy(child, depth + 1)
            }
            hierarchy.append(node)
        return hierarchy

    def print_hierarchy(node, prefix=""):
        # 构造层信息
        executed = "" if node.get("executed", False) else ""
        info = node.get("info", {}) or {}
        output_shape = f"Output Shape: {info.get('output_shape', 'N/A')}" if node.get("executed") else ""
        num_params = f"Params: {info.get('num_params', 0)}"
        trainable = f"Trainable: {info.get('trainable', 'No')}"

        # 打印并保存层信息
        line = f"{prefix}├─ {node['layer_type']}: {node['name']} {executed}"
        output_lines.append(line)
        print(line)

        if output_shape or num_params or trainable:
            line = f"{prefix}│   {output_shape}, {num_params}, {trainable}"
            output_lines.append(line)
            print(line)

        for child in node.get("children", []):
            print_hierarchy(child, prefix + "│   ")

    # 构建并打印层次结构
    hierarchy = build_hierarchy(model)
    output_lines.append("Model Layer Hierarchy with Execution Info:")
    print("Model Layer Hierarchy with Execution Info:")
    for node in hierarchy:
        print_hierarchy(node)

    # 统计总参数信息
    total_params = sum(info["num_params"] for info in executed_layers)
    trainable_params = sum(info["num_params"] for info in executed_layers if info["trainable"] == "Yes")
    non_trainable_params = total_params - trainable_params

    output_lines.append("\n================================================================")
    output_lines.append(f"Total params: {total_params:,}")
    output_lines.append(f"Trainable params: {trainable_params:,}")
    output_lines.append(f"Non-trainable params: {non_trainable_params:,}")
    output_lines.append("================================================================")

    print("\n================================================================")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("================================================================")

    # 保存到文件
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nModel summary saved to {output_file}")


# 对 form_one 特征进行标准化
def scale_form_one(train_data, val_data, test_data):
    scaler = StandardScaler()
    # 提取训练数据中的 form_one 特征

    def extract_features(dataset):
        features = []
        for i, data in enumerate(dataset):  # 添加索引以便定位问题
            if isinstance(data, tuple):  # 如果 data 是元组，取第一个元素
                data = data[0]  # 提取 HeteroData

            if hasattr(data['atoms'], 'form_one'):

                try:
                    features.append(data['atoms'].form_one.numpy())  # 提取为 NumPy 数组
                    #print(data['atoms'].form_one.numpy())
                except Exception as e:
                    print(f"样本索引 {i} 的 'form_one' 特征提取失败：{e}")
                    print(f"data['atoms']: {data['atoms']}")
                    raise
        print('features',len(features))
        return np.vstack(features)  # 堆叠成二维数组

    train_features = extract_features(train_data)
    scaler.fit(train_features)  # 仅用训练集拟合缩放器
    # 定义转换函数
    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].form_one
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].form_one = torch.tensor(scaled_feature, dtype=torch.float)

        return dataset

    # 对所有数据子集进行标准化
    train_data = transform_dataset(train_data, scaler)
    val_data = transform_dataset(val_data, scaler)
    test_data = transform_dataset(test_data, scaler)

    return train_data, val_data, test_data


def scale_form_one_refit(train_data, val_data, test_data, scale_path):
    scaler = StandardScaler()
    # 提取训练数据中的 form_one 特征

    def extract_features(dataset):
        features = []
        for i, data in enumerate(dataset):  # 添加索引以便定位问题
            if isinstance(data, tuple):  # 如果 data 是元组，取第一个元素
                data = data[0]  # 提取 HeteroData

            if hasattr(data['atoms'], 'form_one'):

                try:
                    features.append(data['atoms'].form_one.numpy())  # 提取为 NumPy 数组
                    #print(data['atoms'].form_one.numpy())
                except Exception as e:
                    print(f"样本索引 {i} 的 'form_one' 特征提取失败：{e}")
                    print(f"data['atoms']: {data['atoms']}")
                    raise
        print('features',len(features))
        return np.vstack(features)  # 堆叠成二维数组

    train_features = extract_features(train_data)
    scaler.fit(train_features)  # 仅用训练集拟合缩放器
    with open(scale_path, 'wb') as file:
        pickle.dump(scaler, file)    
    
    # 定义转换函数
    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].form_one
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].form_one = torch.tensor(scaled_feature, dtype=torch.float)

        return dataset

    # 对所有数据子集进行标准化
    train_data = transform_dataset(train_data, scaler)
    val_data = transform_dataset(val_data, scaler)
    test_data = transform_dataset(test_data, scaler)

    return train_data, val_data, test_data

def scale_AA(train_data, val_data, test_data):
    scaler = StandardScaler()
    # 提取训练数据中的 AA 特征

    def extract_features(dataset):
        features = []
        for i, data in enumerate(dataset):  # 添加索引以便定位问题
            if isinstance(data, tuple):  # 如果 data 是元组，取第一个元素
                data = data[0]  # 提取 HeteroData

            if hasattr(data['atoms'], 'AA'):

                try:
                    features.append(data['atoms'].AA.numpy())  # 提取为 NumPy 数组
                    #print(data['atoms'].AA.numpy())
                except Exception as e:
                    print(f"样本索引 {i} 的 'AA' 特征提取失败：{e}")
                    print(f"data['atoms']: {data['atoms']}")
                    raise
        print('features',len(features))
        return np.vstack(features)  # 堆叠成二维数组

    train_features = extract_features(train_data)
    scaler.fit(train_features)  # 仅用训练集拟合缩放器
    # 定义转换函数
    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].AA
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].AA = torch.tensor(scaled_feature, dtype=torch.float)

        return dataset

    # 对所有数据子集进行标准化
    train_data = transform_dataset(train_data, scaler)
    val_data = transform_dataset(val_data, scaler)
    test_data = transform_dataset(test_data, scaler)

    return train_data, val_data, test_data

def scale_AA_refit(train_data, val_data, test_data, scale_path):
    scaler = StandardScaler()
    # 提取训练数据中的 AA 特征

    def extract_features(dataset):
        features = []
        for i, data in enumerate(dataset):  # 添加索引以便定位问题
            if isinstance(data, tuple):  # 如果 data 是元组，取第一个元素
                data = data[0]  # 提取 HeteroData

            if hasattr(data['atoms'], 'AA'):

                try:
                    features.append(data['atoms'].AA.numpy())  # 提取为 NumPy 数组
                    #print(data['atoms'].AA.numpy())
                except Exception as e:
                    print(f"样本索引 {i} 的 'AA' 特征提取失败：{e}")
                    print(f"data['atoms']: {data['atoms']}")
                    raise
        print('features',len(features))
        return np.vstack(features)  # 堆叠成二维数组

    train_features = extract_features(train_data)
    scaler.fit(train_features)  # 仅用训练集拟合缩放器
    with open(scale_path, 'wb') as file:
        pickle.dump(scaler, file)    
    
    # 定义转换函数
    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].AA
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].AA = torch.tensor(scaled_feature, dtype=torch.float)

        return dataset

    # 对所有数据子集进行标准化
    train_data = transform_dataset(train_data, scaler)
    val_data = transform_dataset(val_data, scaler)
    test_data = transform_dataset(test_data, scaler)

    return train_data, val_data, test_data



def scale_form_one_solubility_refit(train_data, val_data, scale_path):
    scaler = StandardScaler()
    # 提取训练数据中的 form_one 特征

    def extract_features(dataset):
        features = []
        for i, data in enumerate(dataset):  # 添加索引以便定位问题
            if isinstance(data, tuple):  # 如果 data 是元组，取第一个元素
                data = data[0]  # 提取 HeteroData

            if hasattr(data['atoms'], 'form_one'):

                try:
                    features.append(data['atoms'].form_one.numpy())  # 提取为 NumPy 数组
                    #print(data['atoms'].form_one.numpy())
                except Exception as e:
                    print(f"样本索引 {i} 的 'form_one' 特征提取失败：{e}")
                    print(f"data['atoms']: {data['atoms']}")
                    raise
        print('features',len(features))
        return np.vstack(features)  # 堆叠成二维数组

    train_features = extract_features(train_data)
    scaler.fit(train_features)  # 仅用训练集拟合缩放器
    with open(scale_path, 'wb') as file:
        pickle.dump(scaler, file)    
    
    # 定义转换函数
    def transform_dataset(dataset, scaler):
        for data in dataset:

            feature_tensor = data[0]['atoms'].form_one
            scaled_feature = scaler.transform(feature_tensor.numpy())
            data[0]['atoms'].form_one = torch.tensor(scaled_feature, dtype=torch.float)

        return dataset

    # 对所有数据子集进行标准化
    train_data = transform_dataset(train_data, scaler)
    val_data = transform_dataset(val_data, scaler)

    return train_data, val_data


class EarlyStopping:
    """ 早停机制：当 val_f1 在 patience 轮内没有改善时，停止训练 """
    def __init__(self, save_path, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_r2 = 0  # 记录最高的验证集 F1-score
        self.best_model_state = None
        self.counter = 0
        self.save_path = save_path

    def __call__(self, val_r2, model, epoch):
        if val_r2 > self.best_r2:  # 发现更好的 F1-score
            self.best_r2 = val_r2
            self.best_model_state = model.state_dict()
            self.counter = 0
            torch.save(model.state_dict(), f"{self.save_path}")  # 保存最优模型
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"🚀 Early stopping triggered after {self.patience} epochs.")
            if self.restore_best_weights:
                model.load_state_dict(self.best_model_state)  # 载入最优模型
            return True  # 停止训练
        return False  # 继续训练

class EarlyStopping_class:
    """ 早停机制：当 val_f1 在 patience 轮内没有改善时，停止训练 """
    def __init__(self, save_path, patience=5, restore_best_weights=True):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_f1 = 0  # 记录最高的验证集 F1-score
        self.best_model_state = None
        self.counter = 0
        self.save_path = save_path

    def __call__(self, val_f1, model, epoch):
        if val_f1 > self.best_f1:  # 发现更好的 F1-score
            self.best_f1 = val_f1
            self.best_model_state = model.state_dict()
            self.counter = 0
            torch.save(model.state_dict(), f"{self.save_path}")  # 保存最优模型
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"🚀 Early stopping triggered after {self.patience} epochs.")
            if self.restore_best_weights:
                model.load_state_dict(self.best_model_state)  # 载入最优模型
            return True  # 停止训练
        return False  # 继续训练

def get_params_with_weight_decay(model, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bn" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    param_groups = []
    
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": float(weight_decay)})

    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return param_groups



def plot_training_curves_refit(train_losses, val_losses, train_accs, val_accs, save_path, epoch):
    """ 绘制训练曲线并保存 """
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.savefig(f"{save_path}")

def plot_training_curves(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, save_path, epoch):
    """ 绘制训练曲线并保存 """
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.plot(test_accs, label='test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.savefig(f"{save_path}")    
