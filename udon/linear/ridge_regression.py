import numpy as np
from udon.linear.regression import Regression


class RidgeRegression(Regression):
  """
  Ridge regression model.

  w* = argmin |t - x @ w| + alpha * |w|_2^2
  """

  def __init__(self, alpha:float=1.):
    self.alpha = alpha

  def fit(self, X:np.array, t:np.ndarray):
    """
    maximum a posteriori estimation of parameter.

    @param X: (N, D) np.ndarray, training data independent variable.
    @param t: (N,) np.ndarray, training data dependent variable.
    """
    eye = np.eye(np.size(X, 1))
    # https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linalg.solve.html
    self.w = np.linalg.solve(self.alpha * eye + X.T @ X, X.T @ t)
  
  def predict(self, X:np.ndarray):
    """
    make prediction given input.
    
    @param X: (N, D) np.ndarray, samples to predict.
    @return (N,) np.ndarray, prediction.
    """
    return X @ self.w



