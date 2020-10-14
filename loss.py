import numpy as np

def cross_entropy_loss(predicitions, desired):
    '''
        Categorical Cross-Entropy Loss function
        Formula: 

        :param predictions: predictions made by network
        :param desired: correct labels
    '''
    return -np.sum(desired * np.log(predicitions))
