import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle

Train = True
n_samples = 100

"""
    x ==> x_poly ==> linearRegression Model ==> y
"""

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, n_samples)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, n_samples)

# x = 2 - 3 * np.random.rand(200,3)                                 # (n_samples, features)
# y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.rand(200,3)     # (n_samples, features)

# transforming the data to include another axis
if len(x.shape) == 1 :
    x = x[:, np.newaxis]    # i.e, will convert (20,) to (20, 1)
    y = y[:, np.newaxis]    # i.e, will convert (20,) to (20, 1)  

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)

def plot(y, x_poly, x=x):
    y_pred = model.predict(x_poly)
    plt.scatter(x, y, s=10)
    plt.scatter(x, y_pred, s=5, color='r')
    plt.show()

class PolynomialRegression:
    def fit(self, model, x_poly, y):
        model.fit(x_poly, y)
        with open('polynomialReg_model.clf', 'wb') as f:
            pickle.dump(model, f)
        plot(y, x_poly)
    
    def predict(self, saved_model, x_poly):
        return saved_model.predict(x_poly)

poly_reg_model = PolynomialRegression()

if Train:
    model = LinearRegression()
    poly_reg_model.fit(model, x_poly, y)

with open('polynomialReg_model.clf', 'rb') as f:
    saved_model = pickle.load(f)

y_pred = poly_reg_model.predict(saved_model, x_poly)
print(y_pred)
print(y_pred.shape)
