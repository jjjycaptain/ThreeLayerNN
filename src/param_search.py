from model import ThreeLayerNet
from train import train,validate
import matplotlib.pyplot as plt
import os
import json
import pickle
import csv

def grid_search(data_train, labels_train, data_val, labels_val):
    # 由于使用了earlystop，所以不用管epoch
    batch_sizes = [16, 32, 64, 128,256]
    learning_rates = [1e-4,5e-4,1e-3,5e-3,1e-2]
    regs = [0.0,1e-4,1e-3,1e-2,1e-1]
    hidden_dims = [(64,64),(128, 128), (256, 256), (512, 512),(1024,1024)]
    lr_decays = [0.8, 0.9, 0.95]
    
    best_val_acc = 0
    best_config = None
    best_model = None
    
    with open('grid_search_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['hd1', 'hd2', 'lr', 'lr_decay', 'batch_size', 'reg', 'val_acc'])

        for (hd1, hd2) in hidden_dims:
            for lr in learning_rates:
                for ld in lr_decays:
                    for bs in batch_sizes:
                        for reg in regs:
                            model_dir = f"./result/ep{100}_bc{bs}_lr{lr}_hda{hd1}_hdb{hd2}_reg{reg}_ld{ld}_relu/"
                            if not os.path.exists(model_dir):
                                model = ThreeLayerNet(input_dim=3072, hidden_dim1=hd1, hidden_dim2=hd2,
                                                      reg=reg)
                                val_acc = train(model, data_train, labels_train, data_val, labels_val,
                                                learning_rate=lr, num_epochs=100, batch_size=bs,
                                                lr_decay=ld)
                            else:
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
                                val_acc, _ = validate(model, data_val, labels_val)
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_config = (hd1, hd2, lr, ld, bs, reg)
                                best_model = model
                            print("hd1=%d hd2=%d lr=%.4f decay=%.2f batch_size=%d reg=%.4f   val_acc=%.4f" %
                                  (hd1, hd2, lr, ld, bs, reg, val_acc))
                            writer.writerow([hd1, hd2, lr, ld, bs, reg, val_acc])

    print("Best val acc: ", best_val_acc)
    print("Best config: hd1=%d hd2=%d lr=%.4f decay=%.2f batch_size=%d reg=%.4f" % best_config)
    return best_model, best_config


def analysis_search(data_train, labels_train, data_val, labels_val):
    search_obj = ['hidden_dim','learning_rate','reg','batch_size']


    # 分析搜索
    for search_obj in search_obj:    
        print(f"Searching for best {search_obj}...")
        lr_decays = [0.8, 0.9, 0.95] if search_obj == 'lr_decay' else [0.95]
        batch_sizes = [16 ,32, 64, 128,256] if search_obj == 'batch_size' else [64]
        learning_rates = [1e-4,5e-4,1e-3,5e-3,1e-2] if search_obj == 'learning_rate' else [1e-3]
        regs = [0.0,1e-4,1e-3,1e-2,1e-1] if search_obj == 'reg' else [1e-3]
        hidden_dims = [(64,64),(128, 128), (256, 256), (512, 512),(1024,1024)] if search_obj == 'hidden_dim' else [(256, 256)]

        best_val_acc = 0
        acc_history = []
        param_history = []

        for (hd1, hd2) in hidden_dims:
            for lr in learning_rates:
                for ld in lr_decays:
                    for bs in batch_sizes:
                        for reg in regs:
                            model_dir = f"./result/ep{100}_bc{bs}_lr{lr}_hda{hd1}_hdb{hd2}_reg{reg}_ld{ld}_relu/"
                            if not os.path.exists(model_dir):
                                model = ThreeLayerNet(input_dim=3072, hidden_dim1=hd1, hidden_dim2=hd2,
                                              reg=reg)
                                _,val_acc = train(model, data_train, labels_train, data_val, labels_val,
                                        learning_rate=lr, num_epochs=100, batch_size=bs,
                                        lr_decay=ld)
                            else:
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
                                val_acc, _ = validate(model, data_val, labels_val)
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                            acc_history.append(val_acc)
                            param_history.append((hd1, hd2, lr, ld, bs, reg))
                            print("hd1=%d hd2=%d lr=%.4f decay=%.2f batch_size=%d reg=%.4f   val_acc=%.4f" %
                              (hd1, hd2, lr, ld, bs, reg, val_acc))

        # 绘制超参数变化和acc变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(acc_history)), acc_history, marker='o', label='Validation Accuracy')

        # 提取只改变的超参数值作为x轴标签
        if search_obj == 'hidden_dim':
            x_labels = [f"{p[0]}-{p[1]}" for p in param_history]
        elif search_obj == 'learning_rate':
            x_labels = [f"{p[2]:.4f}" for p in param_history]
        elif search_obj == 'reg':
            x_labels = [f"{p[5]:.4f}" for p in param_history]
        elif search_obj == 'batch_size':
            x_labels = [f"{p[4]}" for p in param_history]
        else:
            x_labels = [f"{p}" for p in param_history]

        plt.xticks(range(len(x_labels)), x_labels, rotation=45, fontsize=8)
        plt.xlabel(f'Varying {search_obj}')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Validation Acc vs {search_obj}')
        plt.legend()
        plt.tight_layout()

        # 保存曲线到'./analysis/'目录
        os.makedirs('./analysis/', exist_ok=True)
        plt.savefig(f'./analysis/{search_obj}_acc_curve.png')
        plt.close()
