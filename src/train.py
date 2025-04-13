import numpy as np
from data_loader import load_dataset, split_train_valid_set, Dataloader
from model import ThreeLayerNet
import pickle
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

def plot_training_curves(train_losses, val_losses, val_accuracies, model_dir):
    # 训练集loss曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(model_dir + 'train_loss.png')
    plt.close()

    # 验证集loss曲线
    plt.figure()
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.savefig(model_dir + 'val_loss.png')
    plt.close()

    # 验证集ACC曲线
    plt.figure()
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(model_dir + 'val_accuracy.png')
    plt.close()


def validate(model, data, labels, batch_size=256):
    # 分批预测，计算accuracy
    num_samples = data.shape[0]
    batch_num = int(np.ceil(num_samples / batch_size))
    correct = 0
    loss = 0.0
    for i in range(0, num_samples, batch_size):
        data_batch = data[i:i+batch_size]
        labels_batch = labels[i:i+batch_size]
        scores, batch_loss = model.loss(data_batch,labels_batch,valid=True)  # y=None时返回scores
        preds = np.argmax(scores, axis=1)
        correct += np.sum(preds == labels_batch)
        loss += batch_loss
    acc = correct / num_samples
    loss /= batch_num
    return acc, loss

def train(model, data_train, labels_train, data_val, labels_val,
          learning_rate=1e-3, num_epochs=10, batch_size=128,
          lr_decay=0.5, early_stop_patience=5):
    # 构建dataloader
    train_loader = Dataloader(data_train, labels_train, batch_size=batch_size, shuffle=True)
    
    original_lr = learning_rate
    best_val_acc = 0.0
    best_params = None
    train_losses = []
    val_losses = []
    val_accuracies = []
    num_batchs = int(np.ceil(data_train.shape[0] / batch_size))    
    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        # 训练一个epoch
        for batch_data, batch_labels in train_loader:
            loss, grads = model.loss(batch_data, batch_labels)
            train_losses.append(loss)
            
            # 参数更新
            for param_name in model.params:
                model.params[param_name] -= learning_rate * grads[param_name]
            # 每100步打印一次loss
            if len(train_losses) % 100 == 0:
                print(f"Step [{len(train_losses)}/{num_epochs*num_batchs}], Loss: {loss:.4f}")
        
        # 每个 epoch 结束后进行验证
        val_acc, val_loss = validate(model, data_val, labels_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], LR={learning_rate:.6f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 记录最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}
            patience_counter = 0  
        else:
            patience_counter += 1 
        
        # 检查是否需要早停
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
        # 学习率衰减
        learning_rate *= lr_decay
    
    # 恢复最优权重
    model.params = best_params
    # 保存模型权重
    model_dir = f"./result/ep{num_epochs}_bc{batch_size}_lr{original_lr}_hda{model.hidden_dim1}_hdb{model.hidden_dim2}_reg{model.reg}_ld{lr_decay}_{model.activation}/"
    os.makedirs(model_dir, exist_ok=True)
    with open(model_dir+'model.pkl', 'wb') as f:
        pickle.dump(model.params, f)
    # 保存模型参数到config.json文件
    config = {
        'input_dim': model.input_dim,
        'hidden_dim1': model.hidden_dim1,
        'hidden_dim2': model.hidden_dim2,
        'num_classes': model.num_classes,
        'reg': model.reg,
        'activation': model.activation,
        'weight_scale': model.weight_scale,
    }
    with open(model_dir+'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    # 保存训练集loss曲线和验证集loss曲线，验证集ACC曲线的图片
    plot_training_curves(train_losses, val_losses, val_accuracies, model_dir)

    return model,best_val_acc


if __name__ == '__main__':
    # 加载数据集
    print('Loading dataset...')
    root_dir = './dataset/'
    data_train, labels_train, data_test, labels_test = load_dataset(root_dir)
    data_train, labels_train, data_val, labels_val = split_train_valid_set(data_train, labels_train, val_ratio=0.2)
    
    print("Train data shape: ", data_train.shape, "Train labels shape: ", labels_train.shape)
    print("Validation data shape: ", data_val.shape, "Validation labels shape: ", labels_val.shape)
    print("Test data shape: ", data_test.shape, "Test labels shape: ", labels_test.shape)

    # 处理
    print('training...')
    model = ThreeLayerNet(input_dim=3072, hidden_dim1=100, hidden_dim2=100, reg=1e-3)
    best_val_acc = train(model, data_train, labels_train, data_val, labels_val,learning_rate=1e-3, num_epochs=10, batch_size=128, lr_decay=0.8)
    print("Best validation accuracy: ", best_val_acc)

