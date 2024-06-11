
import numpy as np
import scipy.linalg
import math
import numpy as np
from scipy.linalg import block_diag

class MCStFException(Exception):
    """Raise for errors in the MCKF, usually due to bad inputs"""

class MCStF:
    def __init__(self, process_noise, initial_state, initial_covar, epsilon, sigma, iteration, dimension_state_vector, degree_of_freedom):
        self.epsilon = epsilon  # Error limitation
        self.Q = process_noise  # Process noise
        self.x = initial_state  
        self.p = initial_covar  # Error covariance matrix
        self.sigma = sigma      # Kernel bandwith
        self.iter = iteration   # iteration
        self.d = dimension_state_vector # Dimension of state vector
        self.v = degree_of_freedom # Degree of freedom
    def gaussian_kernel(self, error):
        G_sigma = math.exp(- (pow(error, 2)) / (2 * (pow(self.sigma, 2))))
        return G_sigma
    
    
    def prediction(self, x_est, p_est, Q, F):
        x_pred = np.dot(F, x_est)
        p_pred = np.dot(np.dot(F, p_est), F.T ) + Q
        return x_pred, p_pred
    
    
    def uptade(self, x_pred, p_pred, H, Y, R, v, d):
        
        I = np.identity(len(x_pred))   
        x_est_prev = x_pred
        
        x_est = 0
        
        p_est = 0
                 
        for _ in range(self.iter):

            if len(np.shape(R)) > 0:
                Br = np.linalg.cholesky(R)
            else:
                Br = np.array(math.sqrt(R))
                
            Bp = np.linalg.cholesky(p_pred)  # Cholesky Decomposition 
            B = block_diag(Bp, Br)           # Define Diagonal
            D = np.dot(np.linalg.inv(B), np.concatenate([x_pred, Y]))
            W = np.dot(np.linalg.inv(B), np.concatenate([I, H]))

            E = D - np.dot(W, x_est_prev)
            
            Cx_diag = [] 
            Cy_diag = []
            
            for i in range(len(x_pred)):
                Cx_diag.append(self.gaussian_kernel(E[i]))
            
            for i in range(len(x_pred)):
                Cy_diag.append(self.gaussian_kernel(E[i]))
            
            Cx = np.diag(Cx_diag)
            Cy = np.diag(Cy_diag)   
        
            p_pred = np.dot(np.dot(Bp, np.linalg.inv(Cx)), Bp.T)
                            
            R = np.dot(np.dot(Br, np.linalg.inv(Cy)), Br.T)
                       
            err = Y - np.dot(H, p_pred)
            S = (np.dot(H, np.linalg.inv(p_pred)), H.T) + R
            x_star = x_pred + np.dot(np.dot(np.dot(p_pred, H.T), np.linalg.matrix_power(S,-1)), err )
            delta_2 = np.dot(np.dot(np.linalg.inv(err), np.linalg.matrix_power(S,-1)), err)
            p_star = ((v + delta_2)/ (v + d)) * (p_pred - np.dot(np.dot(np.dot(np.dot(p_pred, H.T), np.linalg.matrix_power(S,-1))), H), p_pred)
            v_star = v + delta_2
            x_est = x_star
            
            value = self.compare(x_est_prev, x_est)
            x_est_prev = x_est
            
            if value == True:
                p_est = self.post_cov(v_star, v, p_star)
                break
            else:
                pass
            
        return x_est, p_est
    
    
    def compare(self, x_est_prev, x_est):
        
        res = np.linalg.norm(x_est - x_est_prev) / np.linalg.norm(x_est_prev)
            
        if res <= self.epsilon:
            val = True
        else:
            val = False
        return val
        
    def post_cov(self, v_star, v, p_star):
        p_est = np.dot((v_star/(v_star - 2)), (((v - 2)/v)*p_star))
        return p_est