"""
-----------------------------------------------------------------------
PROJECT: RNN BASED SPELLING CORRECT
MODULE: RNN
AUTHOR: THOMAS HOLLE
    
DESCRIPTION:
Module containing a class defining a simple neural network structure,
and associated helper functions.
-----------------------------------------------------------------------
"""

if __name__ == '__main__':
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('darkgrid')
    np.random.seed(1)
    #define a toy dataset
    """
    ---------------------------------------------------------------------------
    the input data X used in this example consists of 20 binary sequences
    of 10 timesteps each. Each input sequence is generated from a uniform
    random distribution which is rounded to 0 or 1. the output targets, t
    are the number of times '1' occurs in the sequence {which is equal to 
    the sum of that sequence since it is binary}
    ---------------------------------------------------------------------------
    """
    
    num_samples = 20
    sequence_length = 10
    
    #generate the sequences
    X = np.zeros(num_samples,sequence_length)
    for row_index in range(num_samples):
        X[row_index,:] = np.around(np.random.rand(sequence_length)).astype(int)
    #label sequences with targets
    t = np.sum(X,axis=1)
    
    """
    ---------------------------------------------------------------------------
    forward step will unroll the network and compute the forward activations just
    as in regular backpropagation, the final output will be used to compute the
    loss-function from which the error signal used for training will be derived
    when unfolding a RNN over multiple timesteps each layer computes the same
    recurrance relation on different timesteps. this recurrance relation in our
    model is defined in the update_state function.
    
    The forward_states function computes the states over increasing timesteps by
    applying the update_state method in a for-loop. Note that the forward steps
    for multiple sequences can be computed in paralell by use of vectorization.
    since the newtwork begins without seeing anything of the sequence, an initial
    state needs to be provided, in this example, this initial state is set to 0,
    although it is possible to treat this as a hyperparameter.
    
    Finally the loss at the output is computed, in this example, mean squared error
    is used over all sequences in the input data.
    ---------------------------------------------------------------------------
    """
    
    #define the forward step functions...
    def update_state(xk,sk,wx,wRec):
        """
        Compute the state k from the previous state (sk) and current input (xk)
        by use of the input weights (wx) and recursive weights (wRec).
        """
        return xk*wx*sk*wRec
    
    def forward_states(X,wx,wRec):
        """
        Unfold the network and compute all state activations given the input X,
        input weights wx, and recursive weights (wRec). Return the state activations
        in a matrix, the last column S[:-1] contains the final activations.
        """
        #init the matrics to hold states for all input sequences. with initial state set to 0
        S = np.zeros((X.shape[0],X.shape[1]+1))
        
        #use the recurrance relation defined by update_state to update the states through time.
        for k in range(0,X.shape[1]):
            #S[k] = S[k-1] * wRec + X[k] * wx
            S[:,k + 1] = update_state(X[:,k],S[:,k],wx,wRec)
        return S
    
    def loss(y,t):
        """MSE between the targets t and the output activations y"""
        return np.mean((t - y)**2)
    
    """
    ---------------------------------------------------------------------------
    The backward step will begin with computing the gradient of the loss with
    respect to the output of the network \partial E /\partial y using the 
    output_gradient function. This gradient will then be propagated backwards
    through time (layer by layer) from the output to the input to update the
    parametrers by the backward_gradient function.
    ---------------------------------------------------------------------------
    """
    
    def output_gradient(y,t):
        """
        Gradient of the MSE loss function with respect to  the output (y)
        """
        return 2.*(y-t)
    
    def backward_gradient(X,S,grad_out,wRec):
        """
        Backpropagate the gradient computed at the output (grad_out) through
        the network. Accumulate the parameter gradients for wX and wRec for each
        layer by addition. Return the parameter gradients as a tuple, and the
        fradients at the output of each layer.
        """
        
        #initialise the array that stores the gradients of the loss with respect to the states.
        grad_over_time = np.zeros((X.shape[0],X.shape[1]+1))
        grad_over_time[:,-1] = grad_out
        #set gradient accumulators to 0
        wx_grad = 0
        wRec_grad = 0
        for k in range(X.shape[1],0,-1):
            #compute the parameter gradients and accumulate the results
            wx_grad += np.sum(np.mean(grad_over_time[:,k] * X[:,k - 1],axis=0))
            wRec_grad += np.sum(np.mean(grad_over_time[:,k]*S[:,k - 1]),axis=0)
            #compute the gradient at the output of the previous layer
            grad_over_time[:,k-1] = grad_over_time[:,k] * wRec
        return (wx_grad, wRec_grad), grad_over_time
            
#Gradient checking
        
    """
    ---------------------------------------------------------------------------
    we can perdorm gradient checking like in feedforward nets, to assert that
    no mistakes were made while computing the gradients, gradient checking assets
    that the gradient computed by backpropagation is close to the numetical gradient
    ---------------------------------------------------------------------------
    """
    
    
            
            