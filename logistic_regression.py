import numpy as np
import pickle
from generate_datasets import load_dataset

Train = False

x_orig, y_orig = load_dataset()
x = x_orig.T
y = y_orig.reshape(1, y_orig.shape[0])

class logisticRegression:
    def __init__(self, learning_rate=0.05, num_iteration=1000):
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def sigmoid(self, z):
        return 1.0 / (1 + np.e**(-z))
    
    def cost_function(self, y, h_x):
        _, num_examples = y.shape
        cost_1 = - y * np.log(h_x)
        cost_0 = - (1 - y) * np.log(1 - h_x)
        cost = np.sum(cost_1 + cost_0) / num_examples
        return cost

    def fit(self, x, y):
        num_features, num_examples = x.shape
        self._weights = np.random.rand(num_features, 1)

        loss = []
        for i in range(self.num_iteration):
            z = np.dot(self._weights.T, x)
            h_x = self.sigmoid(z)
            L = self.cost_function(y, h_x)
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

logistic_reg_model = logisticRegression(learning_rate=0.2, num_iteration=10000)

if Train:
    model_weights = logistic_reg_model.fit(x, y)
    with open('logistic_reg_model_weight.clf', 'wb') as f:
        pickle.dump({'model_weights':model_weights}, f)

with open('logistic_reg_model_weight.clf', 'rb') as f:
    model_weights = pickle.load(f)

y_pred = logistic_reg_model.predict(x=x, model_weights=model_weights)
print(len(y_pred))