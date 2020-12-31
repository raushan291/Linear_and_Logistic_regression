import numpy as np
import pickle
from generate_datasets import load_dataset

"""
    ### Solving problem of Overfitting ###

    ``` L2 (Ridge regression) Regression ```

    j(theta) = 1/2m * { SUMATION[ (h_theta_i - y_i) ^ 2 ] + lambda * SUMATION[ theta_j ^ 2 ] };

    ``` L1 (Lasso regression) Regression ```

    j(theta) = 1/2m * { SUMATION[ (h_theta_i - y_i) ^ 2 ] + lambda * SUMATION[ |theta_j| ] };

        where, i --> 1 to m,  j --> 1 to n  ;    {m = num_training_data, n = num_features}
"""

Train = False

x_orig, y_orig = load_dataset(path='/home/rakumar/DL_and_ML/binary_digitsDatasets.h5')
x = x_orig.T
y = y_orig.reshape(1, y_orig.shape[0])

class logisticRegression:
    def __init__(self, learning_rate=0.05, regularization_parameter=10, num_iteration=1000):
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.regularization_parameter = regularization_parameter

    def sigmoid(self, z):
        return 1.0 / (1 + np.e**(-z))
    
    def regularized_cost_function(self, y, h_x, weights, regularization_technique='L2'):
        _, num_examples = y.shape
        cost_1 = - y * np.log(h_x)
        cost_0 = - (1 - y) * np.log(1 - h_x)
        cost = np.sum(cost_1 + cost_0) / num_examples
        if regularization_technique == 'L2':
            regularized_cost = cost + (self.regularization_parameter * np.sum(weights ** 2)) / num_examples
        if regularization_technique == 'L1':
            regularized_cost = cost + (self.regularization_parameter * np.sum(np.abs(weights))) / num_examples
        return regularized_cost

    def fit(self, x, y):
        num_features, num_examples = x.shape
        self._weights = np.random.rand(num_features, 1)

        loss = []
        for i in range(self.num_iteration):
            z = np.dot(self._weights.T, x)
            h_x = self.sigmoid(z)
            L = self.regularized_cost_function(y, h_x, self._weights, regularization_technique='L2')
            dL = (1.0/num_examples) * np.dot( x, (h_x - y).T)
            self._weights -= self.learning_rate * dL
            loss.append(L)
            if i % 1 == 0 or i == self.num_iteration-1:
                print('iteration '+str(i)+'  loss: '+str(L))

        y_pred = self.sigmoid(np.dot(self._weights.T, x))
        p = [1 if i > 0.5 else 0 for i in y_pred[0]]
        correct = np.sum(np.asarray(p) == y[0])
        training_accuracy = (correct/num_examples)*100
        print('training_accuracy = '+str(round(training_accuracy, 2))+'% ('+str(correct)+'/'+str(num_examples)+')')
        
        return self._weights

    def predict(self, x, model_weights):
        y_pred = self.sigmoid(np.dot(model_weights['model_weights'].T, x))
        p = [1 if i > 0.5 else 0 for i in y_pred[0]]
        return p

logistic_reg_model = logisticRegression(learning_rate=0.2, regularization_parameter=0.01, num_iteration=10000)

if Train:
    model_weights = logistic_reg_model.fit(x, y)
    with open('regularized_logistic_reg_model_weight.clf', 'wb') as f:
        pickle.dump({'model_weights':model_weights}, f)

with open('regularized_logistic_reg_model_weight.clf', 'rb') as f:
    model_weights = pickle.load(f)

y_pred = logistic_reg_model.predict(x=x, model_weights=model_weights)
print(len(y_pred))