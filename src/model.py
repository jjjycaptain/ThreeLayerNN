import numpy as np
from activation import get_activation_function

def softmax(scores):
    # scores shape: (N, C)
    # 输出 shape: (N, C)
    exps = np.exp(scores - np.max(scores, axis=1, keepdims=True)) # 防止运算时指数溢出
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(probs, y):
    N = probs.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), y])
    loss = np.sum(log_likelihood) / N
    
    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores

class ThreeLayerNet:
    def __init__(self, input_dim=3072, hidden_dim1=100, hidden_dim2=100, num_classes=10,
                 weight_scale=1e-2, reg=0.0, activation='relu'):
        self.params = {}
        self.input_dim = input_dim  
        self.num_classes = num_classes
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.reg = reg
        self.weight_scale = weight_scale
        self.activation = activation
        self.activation_func = get_activation_function(activation)
        
        # 初始化 W1, b1
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim1)
        self.params['b1'] = np.zeros(hidden_dim1)
        
        # 初始化 W2, b2
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim1, hidden_dim2)
        self.params['b2'] = np.zeros(hidden_dim2)
        
        # 初始化 W3, b3
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim2, num_classes)
        self.params['b3'] = np.zeros(num_classes)

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # layer1
        z1 = X.dot(W1) + b1
        a1 = self.activation_func.forward(z1)   # 也可换其它激活
        
        # layer2
        z2 = a1.dot(W2) + b2
        a2 = self.activation_func.forward(z2)
        
        # layer3
        z3 = a2.dot(W3) + b3
        scores = z3   # 最后一层不一定要激活

        # 缓存中间值
        self.cache = (X, z1, a1, z2, a2, z3)
        
        return scores

    def backward(self, dscores):
        X, z1, a1, z2, a2, z3 = self.cache
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        grads = {}
        
        # 反向到 W3, b3
        dW3 = a2.T.dot(dscores)
        db3 = np.sum(dscores, axis=0)
        # 反传到 a2
        da2 = dscores.dot(W3.T)
        
        # a2 = relu(z2)
        dz2 = self.activation_func.backward(da2, z2)
        
        # 反向到 W2, b2
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0)
        # 反传到 a1
        da1 = dz2.dot(W2.T)
        
        # a1 = relu(z1)
        dz1 = self.activation_func.backward(da1, z1)
        
        # 反向到 W1, b1
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)
        
        # 加上正则化项
        dW1 += self.reg * 2 * W1
        dW2 += self.reg * 2 * W2
        dW3 += self.reg * 2 * W3
        
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        
        return grads

    def loss(self, X, y=None,valid=False):
        scores = self.forward(X)
        
        if y is None and not valid:
            # 测试时，直接返回score
            return scores
        
        # 计算 softmax & 交叉熵
        probs = softmax(scores)
        data_loss, dscores = cross_entropy_loss(probs, y)
        
        # L2 正则
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1']**2) +
                                     np.sum(self.params['W2']**2) +
                                     np.sum(self.params['W3']**2))
        loss = data_loss + reg_loss
        
        if valid:
            return scores,loss

        # 反向传播
        grads = self.backward(dscores)
        
        return loss, grads
