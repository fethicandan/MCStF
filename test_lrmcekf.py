
import numpy as np
from numpy.lib.function_base import piecewise
import scipy.linalg
import math
import numpy as np
from scipy.linalg import block_diag
import numdifftools as nd
from lrmcekf import LRMCEKF
import pandas as pd
import matplotlib.pyplot as plt



def state_update(x, Q):
    
    dT = 0.001
    g = 32.2
    ro = 2
    a = 20000
    x = np.unique(x)
    x[0] = x[0] + dT * x[1] + np.random.normal(0, math.sqrt(Q[0,0]))
    x[1] = x[1] + dT * ro * math.exp(-x[0]/a) * (x[1])**2 * (x[2])/2 - dT * g + np.random.normal(0, math.sqrt(Q[1,1]))
    x[2] = x[2] + np.random.normal(0, math.sqrt(Q[2,2]))
    return x


def state_predict(x):
    dT = 0.001
    g = 32.2
    ro = 2.0
    a = 20000.0
    x = np.unique(x)
    # print (type(x[0]), type(x[1]), type(x[2]))
    x[0] =  x[0] + dT * x[1]
    x[1] =  x[1] + dT * ro * math.exp(-x[0]/a) * (x[1])**2 * (x[2])/2 - dT * g
    x[2] =  x[2] 
    return x


def measurement(x_current):
    H_param = 100000
    b = 100000
    y = math.sqrt(b**2 + (x_current[0] - H_param)**2) + np.random.normal(0, 0.1)
    return y


def measurement_pred(x_pred):
    H_param = 100000
    b = 100000
    y_pred = math.sqrt(b**2 + (x_pred[0] - H_param)**2)
    return y_pred


x_est = np.array([300000.0, -20000.0, 0.3e-4], dtype= float)
p_est = block_diag(1e6, 4e6, 1e-4)

x_init = np.array([300000.0, -20000.0, 0.1/100], dtype= float)

fun = lambda x: state_predict(x)
F = nd.Jacobian(fun)

hfun = lambda x: measurement(x)
H = nd.Jacobian(hfun)


Q = block_diag(0.01, 0.01, 0.01)

sigma = 10
epsilon = 1e-3
theta = 0.1

Ru = 0.01 # Initialized mesurement noise covariance matrix
x_current = x_init

x1_plot = []
x2_plot = []
x3_plot = []
x1_est_plot = []
x2_est_plot = []
x3_est_plot = []

lr = LRMCEKF(Q, x_init, p_est, epsilon, sigma, 1000)

for _ in range(1000):
    
    # x_est = x_est.tolist()
    x_pred = state_predict(x_est.tolist())
    p_pred = F(x_est) @ p_est @ F(x_est).T + Q
    
    Y = measurement(x_current)
    y_pred = measurement_pred(x_pred)
    x_est, p_est = lr.update(x_pred, p_pred, y_pred, H(x_pred), Y, Ru)
    
    x_current = state_update(x_current, Q)
        
    x1_plot.append(x_current[0])
    x2_plot.append(x_current[1])
    x3_plot.append(x_current[2])
        
    x1_est_plot.append(x_est[0])
    x2_est_plot.append(x_est[1])
    x3_est_plot.append(x_est[2])


plt.figure()        
plt.plot(x1_plot, label='State')
plt.plot(x1_est_plot, label='Estimate')
plt.legend()

plt.figure()        
plt.plot(x2_plot, label='State')
plt.plot(x2_est_plot, label='Estimate')
plt.legend()

plt.figure()        
plt.plot(x3_plot, label='State')
plt.plot(x3_est_plot, label='Estimate')
plt.legend()

plt.show()      
    
    





"""
    xk1_est = xk1_pred + dT*xk_2 + q1
    xk2_est = xk2_pred + dT*ro*exp(-xk_1, 1/a)*(xk_2)^2*xk_3/2
    xk3_est = xk3_pred + q3
    xk = [xk_1 xk_2 xk_3]^T
"""

