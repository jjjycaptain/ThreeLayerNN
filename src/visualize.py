import numpy as np
import matplotlib.pyplot as plt
import pickle

def visualize_first_layer_weights(W, nrows=4, ncols=4):
    # 选择要查看的神经元索引，比如前 16 个，或随机选 16 个
    num_neurons = nrows * ncols
    
    W = W.T  # 现在 shape 变成 (hidden_dim1, 3072)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))
    
    for i in range(num_neurons):
        ax = axes[i // ncols, i % ncols]
        
        # 取出第 i 个神经元的权重向量: shape (3072,)
        w_i = W[i]
        
        w_i_reshaped = w_i.reshape(3, 32, 32)
        
        # 为了可视化，把 channel-first (3, 32, 32) 转为 channel-last (32, 32, 3)
        w_i_reshaped = np.transpose(w_i_reshaped, (1, 2, 0))
        
        # 做简单归一化：将所有像素映射到 [0, 1]
        w_min, w_max = w_i_reshaped.min(), w_i_reshaped.max()
        if w_max > w_min:  # 避免除以0
            w_i_reshaped = (w_i_reshaped - w_min) / (w_max - w_min)
        else:
            w_i_reshaped = 0.5  # 全部相同的话，就显示灰色
        
        ax.imshow(w_i_reshaped)
        ax.axis('off')
        ax.set_title(f"Neuron {i}")
        
    plt.tight_layout()
    plt.show()
    fig.savefig("./analysis/first_layer_weights_visualization.png")

if __name__ == '__main__':
    model_path = './result/ep100_bc64_lr0.001_hda256_hdb256_reg0.001_ld0.95_relu/model.pkl'
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    W = params['W1']
    visualize_first_layer_weights(W, nrows=4, ncols=4)
