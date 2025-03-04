import numpy as np

# A small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
    def __init__(self):
        self.acc = 0.
        self.loss = 0.

    def forward(self, logit, gt):
        # Stable softmax
        exp_shifted = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        # Save for backward pass
        self.probs = probs
        self.gt = gt

        log_likelihood = -np.log(probs[range(probs.shape[0]), np.argmax(gt, axis=1)] + EPS)
        self.loss = np.sum(log_likelihood) / logit.shape[0]

        self.acc = np.mean(np.argmax(logit, axis=1) == np.argmax(gt, axis=1))
        return self.loss

    def backward(self):
        delta = self.probs
        delta[range(self.gt.shape[0]), np.argmax(self.gt, axis=1)] -= 1
        delta /= self.gt.shape[0]
        return delta