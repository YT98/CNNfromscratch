import numpy as np

def softmax(input):
    '''
        Takes in raw prediction vector and returns probability vector using softmax function.
        Formula: x_i = exp(x_i)/sum(exp(x_i))

        :param input: raw prediction vector
        :returns: probability vector
    '''
    # 
    return np.exp(input)/np.sum(np.exp(input))