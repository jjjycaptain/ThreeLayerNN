from param_search import analysis_search,grid_search
from train import train
from test import test
from data_loader import load_dataset, split_train_valid_set
from model import ThreeLayerNet
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Three Layer Neural Network')
    parser.add_argument('--load_model', action='store_true', help='Load model from the directory')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to save the model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--param_search', type=int, default=0, choices=[0, 1, 2], 
                        help='Parameter search method: 0 for no search, 1 for analysis search, 2 for grid search')

    # 超参数
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay factor')

    # 模型参数
    parser.add_argument('--hidden_dim1', type=int, default=256, help='Hidden layer 1 dimensions')
    parser.add_argument('--hidden_dim2', type=int, default=256, help='Hidden layer 2 dimensions')
    parser.add_argument('--reg', type=float, default=1e-3, help='Regularization strength')
    parser.add_argument('--weight_scale', type=float, default=0.01, help='Weight scale for initialization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function to use')

    return parser.parse_args()

if __name__=='__main__':
    best_val_acc = None
    test_acc = None 

    parser = parse_args()
    data_dir = './dataset'

    if not parser.load_model:
        model = ThreeLayerNet(input_dim=3072, hidden_dim1=parser.hidden_dim1, hidden_dim2=parser.hidden_dim2,
                            num_classes=10, weight_scale=parser.weight_scale, reg=parser.reg, activation=parser.activation)
    
    model_dir = parser.model_dir if parser.model_dir else f"./result/ep{parser.num_epochs}_bc{parser.batch_size}_lr{parser.learning_rate}_hda{parser.hidden_dim1}_hdb{parser.hidden_dim2}_reg{parser.reg}_ld{parser.lr_decay}_{parser.activation}/"

    # 加载数据集
    print('Loading dataset...')
    data_train, labels_train, data_test, labels_test = load_dataset(data_dir)
    data_train, labels_train, data_val, labels_val = split_train_valid_set(data_train, labels_train, val_ratio=0.2)
    print("Train data shape: ", data_train.shape, "Train labels shape: ", labels_train.shape)
    print("Validation data shape: ", data_val.shape, "Validation labels shape: ", labels_val.shape)
    print("Test data shape: ", data_test.shape, "Test labels shape: ", labels_test.shape)

    
    if not parser.param_search:
        # 训练模型
        if parser.train:
            print('\nTraining model...')
            model, best_val_acc = train(model,data_train, labels_train, data_val, labels_val,
                                  learning_rate=parser.learning_rate, num_epochs=parser.num_epochs, batch_size=parser.batch_size,
                                  lr_decay=parser.lr_decay, early_stop_patience=5)
            if parser.test:
                print('\nTesting model...')
                test_acc = test(data_test,labels_test, model, model_dir, load_model=False)

        # 测试模型
        elif parser.test:
            print('\nTesting model...')
            test_acc = test(data_test,labels_test, None, model_dir, load_model=parser.load_model)

        # 保存结果到log.json
        log = {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }
        with open(model_dir+'log.json', 'w') as f:
            json.dump(log, f, indent=4)
    elif parser.param_search == 1:
        # 随机搜索超参数
        print('\nAnalysis search for hyperparameters...')
        analysis_search(data_train, labels_train, data_val, labels_val)
    elif parser.param_search == 2:
        # 网格搜索超参数
        print('\nGrid search for hyperparameters...')
        grid_search(data_train, labels_train, data_val, labels_val)