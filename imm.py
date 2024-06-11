import numpy as np
import scipy.linalg
import math
import numpy as np
from scipy.linalg import block_diag

class IMMException(Exception):
    """Raise for errors in the MCKF, usually due to bad inputs"""

class IMM:
    def __init__(self, length = None, width = None, horizontal = None, vertical=None):
        self.l = length          # Camera footprint
        self.w = width           # Camera footprint
        self.h = horizontal      # Overlapping Percentage among Images  
        self.v = vertical        # Overlapping Percentage among Images
        
    def predicted_model_probability(self):
        for i in range(self.model_num):
            predicted_prob(i,k)=PI(1,i)*Mode_prob(1,k-1)+PI(2,i)*Mode_prob(2,k-1)+PI(3,i)*Mode_prob(3,k-1)
        return predicted_prob
    
    def mixing_weight(self):
                for i=1:Model_number 
            for j=1:Model_number 
                mixing_weight(j,i,k-1)=PI(j,i)*Mode_prob(j,k-1)/Predicted_prob(i,k); 
            end 
        end
        return mixing_weight    
        
    def mixing_covairance(self):
        for i=1:Model_number: 
            Mixing_X(:,i,k-1)=Updated_X(:,1,k-1)*Mixing_weight(1,i,k-1)+Updated_X(:,2,k-1)*Mixing_weight(2,i,k-1)+Updated_X(:,3,k-1)*Mixing_weight(3,i,k-1); 
            Mixing_P(:,:,i,k-1)=(Updated_P(:,:,1,k-1)+(Mixing_X(:,i,k-1)-Updated_X(:,1,k-1))*(Mixing_X(:,i,k-1)-Updated_X(:,1,k-1))')*Mixing_weight(1,i,k-1); 
            Mixing_P(:,:,i,k-1)=Mixing_P(:,:,i,k-1)+(Updated_P(:,:,2,k-1)+(Mixing_X(:,i,k-1)-Updated_X(:,2,k-1))*(Mixing_X(:,i,k-1)-Updated_X(:,2,k-1))')*Mixing_weight(2,i,k-1); 
            Mixing_P(:,:,i,k-1)=Mixing_P(:,:,i,k-1)+(Updated_P(:,:,3,k-1)+(Mixing_X(:,i,k-1)-Updated_X(:,3,k-1))*(Mixing_X(:,i,k-1)-Updated_X(:,3,k-1))')*Mixing_weight(3,i,k-1); 

    def filter(self):
        Measurement_z = measure(:,k,n);
        [Updated_X(:,i,k), Updated_P(:,:,i,k)] = MCKF(Mixing_X(:,i,k-1), Mixing_P(:,:,i,k-1), Q(:,:,i), F(:,:,i), R, H, Measurement_z);

        predicted_z(:,i,k)=H*Updated_X(:,i,k);
        Measurement_res_z(:,i,k)=Measurement_z - predicted_z(:,i,k);
        Residual_S(:,:,i,k)=H*Updated_P(:,:,i,k)*H'+R;
        Model_like_L(i,k)=1e-99+exp(-Measurement_res_z(:,i,k)'*inv(Residual_S(:,:,i,k))*Measurement_res_z(:,i,k)/2)/(det(2*pi*Residual_S(:,:,i,k)))^0.5; 
        
    def update(self):
        mu = Predicted_prob(1,k)*Model_like_L(1,k)+Predicted_prob(2,k)*Model_like_L(2,k)+Predicted_prob(3,k)*Model_like_L(3,k);
        
        for i=1:Model_number 
            Mode_prob(i,k)=Predicted_prob(i,k)*Model_like_L(i,k)/mu; 
        end
        
        Overall_X(:,k)=Updated_X(:,1,k)*Mode_prob(1,k)+Updated_X(:,2,k)*Mode_prob(2,k)+Updated_X(:,3,k)*Mode_prob(3,k); 
        Overall_P(:,:,k)=(Updated_P(:,:,1,k)+(Overall_X(:,k)-Updated_X(:,1,k))*(Overall_X(:,k)-Updated_X(:,1,k))')*Mode_prob(1,k); 
        Overall_P(:,:,k)=Overall_P(:,:,k)+(Updated_P(:,:,2,k)+(Overall_X(:,k)-Updated_X(:,2,k))*(Overall_X(:,k)-Updated_X(:,2,k))')*Mode_prob(2,k); 
        Overall_P(:,:,k)=Overall_P(:,:,k)+(Updated_P(:,:,3,k)+(Overall_X(:,k)-Updated_X(:,3,k))*(Overall_X(:,k)-Updated_X(:,3,k))')*Mode_prob(3,k); 
        