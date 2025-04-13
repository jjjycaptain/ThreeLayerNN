from train import ThreeLayerNet
from data_loader import load_dataset,  Dataloader
import numpy as np
import pickle
import json

def evaluate_accuracy(model, data, labels, batch_size=256):
    # 分批预测，计算accuracy
    num_samples = data.shape[0]
    batch_num = int(np.ceil(num_samples / batch_size))
    correct = 0
    loss = 0.0
    for i in range(0, num_samples, batch_size):
        data_batch = data[i:i+batch_size]
        labels_batch = labels[i:i+batch_size]
        scores = model.loss(data_batch)  # y=None时返回scores
        preds = np.argmax(scores, axis=1)
        correct += np.sum(preds == labels_batch)
    acc = correct / num_samples
    return acc

def test(data_test, labels_test, model, model_dir, load_model=True):
    if load_model:
        print(f"Loading model from {model_dir}...")
        model_path = model_dir + 'model.pkl'
        config_path = model_dir + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        model = ThreeLayerNet(input_dim=config['input_dim'],
                              hidden_dim1=config['hidden_dim1'],
                              hidden_dim2=config['hidden_dim2'],
                              num_classes=config['num_classes'],
                              weight_scale=config['weight_scale'],
                              reg=config['reg'],
                              activation=config['activation'])

        with open(model_path, 'rb') as f:
            best_params = pickle.load(f)
        model.params = best_params
    else:
        print("Using model directly from former training...")
    test_acc = evaluate_accuracy(model, data_test, labels_test)
    print("Test Accuracy: ", test_acc)
    return test_acc


if __name__ == "__main__":
    # 加载数据集
    print('Loading dataset...')
    root_dir = './dataset/'
    model_dir = './result/ep10_bc128_lr0.001_hd1100_hd2100_reg0.001_ld0.8/'
    _, _, data_test, labels_test = load_dataset(root_dir)
    test(data_test, labels_test, None,model_dir , load_model=True)