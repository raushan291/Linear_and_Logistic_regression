import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

"""
    ''' x ==> linearRegression Model ==> y ''' 

    X = (num_features, num_examples) 
    Weights = (num_features, 1)
    Y = (1, num_examples)
    h_x = Weights.T * X     ==>  (1, num_examples) 
    grad_vector = 1/m * (h_x - Y).X  i.e,    [ 1/m * x.(h_x - y).T ] ==> (num_features, 1)
"""

np.random.seed(0)

X = np.random.rand(1,100)
Y = 2 + 3*X + np.random.rand(1, 100)

Train = True

def plot(x, y, model_weight, cost):
    y_pred = np.dot(model_weight.T, x)

    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
    fig.suptitle('Results')

    # data
    ax1.scatter(x, y, s=4)

    # data with prediction line
    ax2.scatter(x, y, s=4)
    ax2.scatter(x, y_pred, color='red', s=5)

    # cost
    cost_x_coord = [ i for i in range(len(cost))]
    ax3.scatter(cost_x_coord, cost, color='pink', s=5)
    plt.show()

class LinearRegressionUsingGD:
    def __init__(self, learning_rate=0.05, num_iteration=1000):
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration

    def fit(self, x, y):
        num_features, num_examples = x.shape
        self._weight = np.random.rand(num_features, 1)
        self.cost = []

        for i in range(self.num_iteration):
            h_x = np.dot(self._weight.T, x)
            grad_vector = (1.0/num_examples) * np.dot(x, (h_x - y).T)
            self._weight -= self.learning_rate * grad_vector

            cost = np.sum((h_x - y)**2) / (2*num_examples)
            self.cost.append(cost)
            if i % 10000 == 0 or i == self.num_iteration-1:
                print('iteration '+str(i)+' ', cost)
        
        return self._weight, self.cost

    def predict(self, x, model_weight):
        return np.dot(model_weight['model_weight'].T, x)

lr_model = LinearRegressionUsingGD(learning_rate=0.001, num_iteration=100000)

if Train:
    model_weight, cost = lr_model.fit(x=X, y=Y)
    with open('model_weight.clf', 'wb') as f:
        pickle.dump({'model_weight':model_weight}, f)
    plot(X, Y, model_weight, cost)

with open('model_weight.clf', 'rb') as f:
    model_weight = pickle.load(f)

print(model_weight)
pred = lr_model.predict(x=X, model_weight=model_weight)
# print('PREDICT = ', pred)
print(pred.shape)
