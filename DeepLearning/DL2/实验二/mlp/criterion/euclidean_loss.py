import numpy as np

class EuclideanLossLayer():
    def __init__(self):
        self.acc = 0.  
        self.loss = 0.

    def forward(self, logit, gt):
        self.logit = logit  
        self.gt = gt        
        self.loss = np.sum((logit - gt) ** 2) / (2 * logit.shape[0])
        # 计算准确度并保存
        correct_preds = np.argmax(logit, axis=1) == np.argmax(gt, axis=1)
        self.acc = np.mean(correct_preds)
        return self.loss

    def backward(self):
        delta = (self.logit - self.gt) / self.logit.shape[0]
        return delta
