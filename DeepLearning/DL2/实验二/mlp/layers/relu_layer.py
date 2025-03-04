import numpy as np

class ReLULayer():
    def __init__(self):
        self.trainable = False # 没有可训练的参数

    def forward(self, Input):
        ############################################################################
        # TODO: 
        # 对输入应用ReLU激活函数并返回结果
        self.output = np.maximum(0, Input)
        return self.output
        ############################################################################

    def backward(self, delta):
        ############################################################################
        # TODO: 
        # 根据delta计算梯度
        grad_input = delta * (self.output > 0)
        return grad_input
        ############################################################################