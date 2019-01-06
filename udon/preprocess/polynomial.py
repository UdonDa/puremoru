import itertools
import functools
import numpy as np


class PolynomialFeatures(object):
  """
    Transforms input array with polynomial features.
      sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    
    ======
    x = [
      [a, b],
      [c, d]]
    y = PolunomialFeatures.transform(x)
    y = [
      [1, a, b, a^2, a*b, b^2],
      [1, c, d, c^2, c*d, d^2]]
  """

  def __init__(self, degree=2):
    """
    @param degree: int, degree of polynomial.
    """
    assert isinstance(degree, int)
    self.degree = degree

  def transform(self, x):
    """
    @param x: (sample_size, n) ndarray, input.
    @return y: (sanple_size, 1 + nC1 + ... + nCd) ndarray, polynomial features.
    """
    if x.ndim == 1:
      x = x[:, None]
    x_t = x.transpose()
    poly_features = [np.ones(len(x))]
    for degree in range(1, self.degree+1):
      for items in itertools.combinations_with_replacement(x_t, degree):
        poly_features.append(functools.reduce(lambda x, y: x*y, items))
    y = np.asarray(poly_features).transpose()
    return y