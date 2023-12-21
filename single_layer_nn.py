import numpy as np
import copy

class NN_model:
    def __init__(self):
        self.parameters = {}

    def layer_sizes(self,X,Y):
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
        return(n_x,n_h,n_y)
    
    def sigmoid(self,x):
        z = 1/(1+np.exp(-x))
        return z

    def initialize_params(self,n_x,n_h,n_y):
        np.random.seed(2)
        W1 = np.random.randn(n_h,n_x) *0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) *0.01
        b2 = np.zeros((n_y,1))

        parameters = { "W1": W1,
                    "W2": W2,
                    "b1":b1,
                    "b2":b2
                    }

        return parameters
    
    def forward_prop(self,X,parameters):

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2,A1) + b2
        A2 = self.sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))

        cache = { "Z1": Z1,
                    "Z2": Z2,
                    "A1":A1,
                    "A2":A2
                    }
        
        return A2,cache
    
    
    def compute_cost(self,A2, Y):
        m = Y.shape[1]

        # Compute the cross-entropy loss
        logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
        cost = -np.sum(logprobs)/m

        # Ensure cost is a scalar (convert to float and squeeze)
        cost = float(np.squeeze(cost))

        return cost
    
    def back_prop(self,parameters,Y,X,cache):

        m = Y.shape[1]

        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2,A1.T)/m
        db2 = (np.sum(dZ2,axis=1,keepdims=True))/m
        dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
        dW1 = np.dot(dZ1,X.T)/m
        db1 = (np.sum(dZ1,axis=1,keepdims=True))/m

        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads
    
    def update_parameters(self,parameters, grads, learning_rate = 1.2):
    
   
        W1 = copy.deepcopy(parameters["W1"])
        b1 = parameters["b1"]
        W2 = copy.deepcopy(parameters["W2"])
        b2 = parameters["b2"]
        
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]
    
        W1 = W1 - learning_rate*dW1
        b1 = b1 - learning_rate*db1
        W2 = W2 - learning_rate*dW2
        b2 = b2 - learning_rate*db2
        
        
        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        
        return parameters


    
    def fit(self, X, Y, n_h, epochs):
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]

        self.parameters = self.initialize_params(n_x, n_h, n_y)

        for i in range(epochs):
            A2, cache = self.forward_prop(X, self.parameters)
            cost = self.compute_cost(A2, Y)
            grads = self.back_prop(self.parameters, Y, X, cache)
            self.parameters = self.update_parameters(self.parameters, grads)

            if i % 10000 == 0:
                print (f"The cost after {i} iterations is : {cost}")

        return self.parameters
    
    def predict(self,X):
        if not self.parameters:
            raise ValueError("Model parameters are not initialized. Please call fit() first.")
        
        A2 , cache = self.forward_prop(X,self.parameters)
        predictions = (A2 > 0.5)

        return predictions
        


    