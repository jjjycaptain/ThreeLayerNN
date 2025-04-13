import numpy as np

class ReLU:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, dout, z):
        dz = dout * (z > 0)
        return dz
    
class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, dout, z):
        sig = self.forward(z)
        dz = dout * sig * (1 - sig)
        return dz

class Tanh:
    def forward(self, z):
        return np.tanh(z)

    def backward(self, dout, z):
        tanh = self.forward(z)
        dz = dout * (1 - tanh ** 2)
        return dz

def get_activation_function(name):
    if name.lower() == "relu":
        return ReLU()
    elif name.lower() == "sigmoid":
        return Sigmoid()
    elif name.lower() == "tanh":
        return Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")