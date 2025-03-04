import numpy as np
 
class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        # Initialize velocity for momentum
        self.v_W = np.zeros_like(model.W)
        self.v_b = np.zeros_like(model.b)

    def step(self):
        if self.model.trainable:
            # Momentum update
            self.v_W = self.momentum * self.v_W - self.learning_rate * self.model.grad_W
            self.v_b = self.momentum * self.v_b - self.learning_rate * self.model.grad_b
            
            # Apply update
            self.model.W += self.v_W
            self.model.b += self.v_b
 
                # # Weight update without momentum
                # layer.W += -self.learning_rate * layer.grad_W
                # layer.b += -self.learning_rate * layer.grad_b
 
                ############################################################################

