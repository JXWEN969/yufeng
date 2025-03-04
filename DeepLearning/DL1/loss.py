import numpy as np

# A small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLoss:
    """
    Softmax Cross-Entropy Loss layer
    """

    def __init__(self, num_input, num_output, trainable=True):
        """
        Initialize SoftmaxCrossEntropyLoss instance
        Args:
            num_input: Size of each input sample
            num_output: Size of each output sample
            trainable: Whether if this layer is trainable
        """
        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def forward(self, Input, labels):
        """
        Forward pass of SoftmaxCrossEntropyLoss layer
        Args:
            Input: Input data (batch_size, num_input)
            labels: Ground truth label (batch_size,)
        Returns:
            loss: Cross-entropy loss
            acc: Accuracy
        """
        # Linear transformation
        scores = np.dot(Input, self.W) + self.b  # Shape: (batch_size, num_output)
        # Softmax probabilities
        scores_exp = np.exp(scores)
        scores_sum = np.sum(scores_exp, axis=1, keepdims=True)
        probs = scores_exp / scores_sum  # Shape: (batch_size, num_output)
        # Cross-entropy loss
        N = Input.shape[0]  # Batch size
        C = self.num_output  # Number of classes
        # Convert labels to one-hot vectors
        y_onehot = np.zeros((N, C))
        y_onehot[np.arange(N), labels] = 1  # Shape: (batch_size, num_output)
        # Compute loss
        loss = -np.sum(y_onehot * np.log(probs)) / N
        # Compute accuracy
        preds = np.argmax(probs, axis=1)  # Shape: (batch_size,)
        acc = np.mean(preds == labels)
        # Save some arrays for gradient computing
        self.Input = Input
        self.probs = probs
        self.y_onehot = y_onehot
        return loss, acc

    def gradient_computing(self):
        """
        Compute gradients of SoftmaxCrossEntropyLoss
        """
        # Gradient of loss with respect to scores
        d_scores = (self.probs - self.y_onehot) / self.Input.shape[0]  # Shape: (batch_size, num_output)
        # Gradient of loss with respect to W
        self.grad_W = np.dot(self.Input.T, d_scores)  # Shape: (num_input, num_output)
        # Gradient of loss with respect to b
        self.grad_b = np.sum(d_scores, axis=0, keepdims=True)  # Shape: (1, num_output)

    def XavierInit(self):
        """
        Initialize weights using Xavier initialization
        """
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
