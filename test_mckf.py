#!/usr/bin/env python

from mckf import MCKF
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import time
import pandas as pd

"""Example 1"""

theta = math.pi/18

F = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])

H = np.array([[1, 1]])

x_real = np.array([[0, 1]]).T # x real initial state

x_est = x_real + np.array([[1, 1]]).T * np.random.normal(0, 0.1) # estimated initial state (adding noise)

p_est = 0.01 * np.diag([1,1]) # estimated initial error covariance

y_real = np.dot(H, x_real) + np.random.normal(0, 0.1)*0.8 + np.random.gumbel(0, 0.1)*0.2  # + np.random.normal(0, 10)*0.1 

Q = np.array([[0.01, 0], [0, 0.01]]) # Process noise

sigma = 10

epsilon = 1e-3

R = 0.01 # Initialized mesurement noise covariance matrix

# MCKF (process_noise, initial_state, initial_covar, epsilon, sigma, iteration)
state_estimate = MCKF(Q, x_est, p_est, epsilon, sigma, 1000)

x1_plot = []
x2_plot = []
x1_est_plot = []
x2_est_plot = []

diff1_plot = []
diff2_plot = []

for _ in range(1000):

    x_pred, p_pred = state_estimate.prediction(x_est, p_est, Q, F)
    
    x_real = np.dot(F, x_real) + np.reshape(np.random.normal(0, 0.1, 2), (2,1))
    
    y_real = np.dot(H, x_real) + np.random.normal(0, 0.1)*0.9 + np.random.gumbel(0, 0.1)*0.2  # np.random.normal(0, 10)*0.1
    
    x_est, p_est = state_estimate.uptade(x_pred, p_pred, H, y_real, R)
    
    print ("--------------------------------------------------------")
    print ("Real state: ", x_real)
    print ("Estimated state: ", x_est)
    print ("Difference: ", x_real - x_est)
        
    x1_plot.append(x_real[0, 0])
    x2_plot.append(x_real[1, 0])
        
    x1_est_plot.append(x_est[0, 0])
    x2_est_plot.append(x_est[1, 0])

    diff1_plot.append(x_real[0, 0] - x_est[0, 0])
    diff2_plot.append(x_real[1, 0] - x_est[1, 0])    

d1 = pd.DataFrame(x1_plot)
d2 = pd.DataFrame(x2_plot)
d3 = pd.DataFrame(x1_est_plot)
d4 = pd.DataFrame(x2_est_plot)
dif1 = pd.DataFrame(diff1_plot)
dif2 = pd.DataFrame(diff2_plot)

d1.to_csv('d1.txt', index=False, header=False)
d2.to_csv('d2.txt', index=False, header=False)
d3.to_csv('d3.txt', index=False, header=False)
d4.to_csv('d4.txt', index=False, header=False)
dif1.to_csv('dif3.txt', index=False, header=False)
dif2.to_csv('dif4.txt', index=False, header=False)


    
plt.figure()        
plt.plot(x1_plot)
plt.plot(x1_est_plot)

plt.figure()        
plt.plot(x2_plot)
plt.plot(x2_est_plot)

plt.figure()        
plt.plot(diff1_plot)

plt.figure()        
plt.plot(diff2_plot)

plt.show()  
    
    
    