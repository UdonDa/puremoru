import numpy as np
from udon.linear.regression import Regression


class LinearRegression(Regression):
  """
  Linear regression model.
    y = X @ w
    t ~ N(t|X @ w, var)
  """
  def fit(self, X:np.ndarray, t:np.ndarray):
    """
    perform least squares fitting.

    @param X: (N, D) np.ndarray, training independent variable.
    @param t: (N, ) np.ndarray, training dependent variable.
    """
    self.w = np.linalg.pinv(X) @ t
    self.var = np.mean(np.square(X @ self.w  -t ))
  
  def predict(self, X:np.ndarray, return_std:bool=False):
    """
    make prediction given input.
    
    @param X: (N, D) np.ndarray, samples to predict.
    @param return_std: bool, returns standard deviation of eac prediction.

    @return y: (N,) np.ndarray, prediction of each sample.
    @return y_std: (N,) np.ndarray, stadard deviation of each predition.
    """
    y = X @ self.w
    if return_std:
      y_std = np.sqrt(self.var) + np.zeros_like(y)
      return y, y_std
    return y