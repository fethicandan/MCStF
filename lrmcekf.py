
import numpy as np
import scipy.linalg
import math
import numpy as np
from scipy.linalg import block_diag
import numdifftools as nd

class LRMCEKFException(Exception):
    """Raise for errors in the LRMCEKF, usually due to bad inputs"""

class LRMCEKF:
    def __init__(self, process_noise, initial_state, initial_covar, epsilon, sigma, iteration):
        self.epsilon = epsilon  # Error limitation
        self.Q = process_noise  # Process noise
        self.x = initial_state  
        self.p = initial_covar  # Error covariance matrix
        self.sigma = sigma      # Kernel bandwith
        self.iter = iteration   # iteration
        
        
    def gaussian_kernel(self, error):
        G_sigma = math.exp(- (pow(error, 2)) / (2 * (pow(self.sigma, 2))))
        return G_sigma
       
    
    def prediction(self, p_est, Q, F):
        p_pred = np.dot(np.dot(F, p_est), F.T ) + Q
        return p_pred
    
    
    def update(self, x_pred, p_pred, y_pred, H, Y, Ra):
        x_pred = np.array(x_pred)
        p_pred = np.array(p_pred)
        y_pred = np.array(y_pred)
        Y = np.array(Y)
        H = np.array(H)
        
        I = np.identity(len(x_pred))   
        x_est_prev = x_pred
        
        x_est = 0
        p_est = 0
        
        psi = Y - y_pred
        eta  = np.dot(np.dot(H, p_pred), H.T) + Ra
        beta = np.dot(np.dot(psi.T, np.linalg.inv(eta)),psi) 
                
        if abs(beta) > 4: #( beta = 0 -> Positive Threshold)
            x_est = x_pred
            p_est = p_pred
        else:
            x_est = x_pred
                 
            # for _ in range(self.iter):
            while True:   
                if len(np.shape(Ra)) > 0:
                    Mr = np.linalg.cholesky(Ra)  #rk values
                else:
                    Mr = np.array(math.sqrt(Ra))
                    
                Mp = np.linalg.cholesky(p_pred)  # Cholesky Decomposition 
                M = block_diag(Mp, Mr)           # Define Diagonal   
                D = np.dot(np.linalg.inv(M), np.concatenate([I, H]))
                z = np.dot(np.linalg.inv(M),np.concatenate([x_pred, (Y - y_pred + np.dot(H, x_pred))]))
                E = z - np.dot(D, x_est)
                    
                Cx_diag = [] 
                Cy_diag = []
                    
                for i in range(len(x_pred)):
                    Cx_diag.append(self.gaussian_kernel(E[i]))
                    
                for i in range(len(x_pred),len(E)):
                    Cy_diag.append(self.gaussian_kernel(E[i]))
                    
                Cx = np.diag(Cx_diag)
                Cy = np.diag(Cy_diag)                   

                p_pred = np.dot(np.dot(Mp, np.linalg.inv(Cx)), Mp.T)        
                R = np.dot(np.dot(Mr, np.linalg.inv(Cy)), Mr.T)   
                K = np.dot(np.dot(p_pred, H.T), np.linalg.inv(np.dot((np.dot(H, p_pred)), H.T) + R ) )
                    
                x_est = x_pred + np.dot(K, (Y - y_pred))
                p_est = self.post_cov(I, K, H, p_pred, R)     
                    
                value = self.compare(x_est_prev, x_est)
                x_est_prev = x_est
                
                if value == True:
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
        
    def post_cov(self, I, K, H, p_pred, R):

        p_est = np.dot((np.dot(I - np.dot(K, H), p_pred)), (I - np.dot(K, H).T )) +  np.dot(np.dot(K, R), K.T)
        return p_est