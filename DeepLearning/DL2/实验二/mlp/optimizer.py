import numpy as np

class SGD():
    def __init__(self, learningRate, weightDecay):
        self.learningRate = learningRate
        self.weightDecay = weightDecay

    # 一步反向传播，逐层更新参数
    def step(self, model):
        layers = model.layerList
        for layer in layers:
            if layer.trainable:
                ############################################################################
                # TODO:
                # 使用layer.grad_W和layer.grad_b计算diff_W and diff_b.
                # 注意weightDecay项.

                diff_W = -self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
                diff_b = -self.learningRate * (layer.grad_b + self.weightDecay * layer.b)
                ############################################################################

                # Weight update
                layer.W += diff_W
                layer.b += diff_b