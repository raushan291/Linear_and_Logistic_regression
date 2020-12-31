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

    def prediction_on_each_class(self, classes, num_samples, labels, trained_parameters):
        for c in classes:
            y = labels[str(c)]
            for i in range(y.shape[1]):
                if y[0][i] == c:
                    y[0][i] = 1
                else:
                    y[0][i] = 0

            weights = trained_parameters[c]
            correct_predictions = 0 
            for i in range(num_samples):
                z = np.dot(weights.T, x)
                h_x = self.sigmoid(z)
                if np.logical_and(h_x[0][i] >= 0.5, y[0][i] == 1):
                    correct_predictions+=1
                if np.logical_and(h_x[0][i] < 0.5, y[0][i] == 0):
                    correct_predictions+=1
            acc = (correct_predictions/num_samples)*100 if correct_predictions != 0 else 0
            print('accuracy for dataset '+str(c)+" = "+str(round(acc, 2))+'% ('+str(correct_predictions)+'/'+str(num_samples)+')')

    def fit(self, x, y):
        num_samples = x.shape[1]
        y_original = y
        classes = np.unique(y)
        trained_parameters = []
        cost_values = []

        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9 = y.copy(), y.copy(), y.copy(), y.copy(), y.copy(), y.copy(), y.copy(), y.copy(), y.copy(), y.copy()
        labels = {'0':y0, '1':y1, '2':y2, '3':y3, '4':y4, '5':y5, '6':y6, '7':y7, '8':y8, '9':y9}

        for c in classes:
            y = labels[str(c)]
            for i in range(y.shape[1]):
                if y[0][i] == c:
                    y[0][i] = 1
                else:
                    y[0][i] = 0

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
            cost_values.append(loss)
            trained_parameters.append(self._weights)

        # classifier predictions
        accuratepredicts = 0                
        for i in range(num_samples):
            probabilities = []
            for j in range(len(classes)):
                weights = trained_parameters[j]
                z = np.dot(weights.T, x[:,i])
                h_x = self.sigmoid(z)
                probabilities.append(h_x)
            
            predict = probabilities.index(max(probabilities))
            if y_original[0][i] == predict:
                accuratepredicts+=1
        acc = (accuratepredicts/num_samples)*100
        print('accuracy = '+str(round(acc, 2))+'%'+' ('+str(accuratepredicts)+'/'+str(num_samples)+')')

        # predictions for each class:
        self.prediction_on_each_class(classes, num_samples, labels, trained_parameters)

        return trained_parameters

    def predict(self, x, num_classes, model_weights):
        num_samples = x.shape[1]
        pred = []
        for i in range(num_samples):
            probabilities = []
            for j in range(num_classes):
                weights = model_weights['model_weights'][j]
                z = np.dot(weights.T, x[:,i])
                h_x = self.sigmoid(z)
                probabilities.append(h_x)
            
            predict = probabilities.index(max(probabilities))
            pred.append(predict)
        return pred

logistic_reg_model = logisticRegression(learning_rate=0.2, num_iteration=2000)

if Train:
    model_weights = logistic_reg_model.fit(x, y)
    with open('multiclass_logistic_reg_model_weight.clf', 'wb') as f:
        pickle.dump({'model_weights':model_weights}, f)

with open('multiclass_logistic_reg_model_weight.clf', 'rb') as f:
    model_weights = pickle.load(f)

test_datasets, y = load_dataset(path='/home/rakumar/DL_and_ML/test_digitsDatasets.h5')
y_pred = logistic_reg_model.predict(x=test_datasets.T, num_classes=3, model_weights=model_weights)
print(y_pred)