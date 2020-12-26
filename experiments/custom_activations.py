from jax import numpy as np

def leaky_tanh(x, alpha=1.0, beta=0.1):
    return np.tanh(alpha*x) + beta*x

def smooth_leaky_relu(x, alpha=1.0):
  r"""Custom activation function
  Source:
  https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function
  """
  return alpha*x + (1 - alpha)*np.logaddexp(x, 0)

def elu(x, alpha=1.0):
  r"""Exponential linear unit activation function.

  Computes the element-wise function:

  .. math::
    \mathrm{elu}(x) = \begin{cases}
      x, & x > 0\\
      \alpha \exp(x - 1), & x \le 0
    \end{cases}
  """
  safe_x = np.where(x > 0, 0., x)
  return np.where(x > 0, x, alpha * np.expm1(safe_x))

def softplus(x):
  r"""Softplus activation function.

  Computes the element-wise function

  .. math::
    \mathrm{softplus}(x) = \log(1 + e^x)
  """
  return np.logaddexp(x, 0)

def get_activation_fn(fn_name, alpha):

  if fn_name == "smooth_leaky_relu":
    return lambda x: smooth_leaky_relu(x, alpha=alpha)
  elif fn_name == "elu":
    return lambda x: elu(x, alpha=alpha)
    
  return softplus
