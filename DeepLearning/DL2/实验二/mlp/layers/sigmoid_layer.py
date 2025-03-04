import numpy as np

class SigmoidLayer():
    def __init__(self):
        self.trainable = False

    def forward(self, Input):
        ############################################################################
        # TODO: 
        # 对输入应用Sigmoid激活函数并返回结果
        self.output = 1 / (1 + np.exp(-Input))
        return self.output
        ############################################################################

    def backward(self, delta):
        ############################################################################
        # TODO: 
        # 根据delta计算梯度
        grad_input = delta * self.output * (1 - self.output)
        return grad_input
        ############################################################################