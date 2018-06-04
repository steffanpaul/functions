from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sb

import tensorflow as tf

#sequence generator function
def seq_generator(num_data, seq_length, dims, seed):
    
    np.random.seed(seed)
    Xsim = np.zeros((num_data, seq_length, 1, 4), np.float32)
    for d in range(num_data):
        Xsim_key = np.random.choice([0,1,2,3], seq_length, [0.25, 0.25, 0.25, 0.25])
        Xsim_hp = np.zeros((seq_length,1, 4))
        for (idx,nuc) in enumerate(Xsim_key):
            Xsim_hp[idx][0][nuc] = 1
        Xsim[d] = Xsim_hp
    return Xsim

    #First order mutagenesis function              
def mutate(sequence, seq_length, dims):
    import numpy as np
    num_mutations = seq_length * dims
    hotplot_mutations = np.zeros((num_mutations,seq_length,1,dims)) 

    for position in range(seq_length):
        for nuc in range(dims):
            mut_seq = np.copy(sequence)          
            mut_seq[0, position, 0, :] = np.zeros(dims)
            mut_seq[0, position, 0, nuc] = 1.0
            
            hotplot_mutations[(position*dims)+nuc] = mut_seq
    return hotplot_mutations



#def secondorder_mutate(X):
def double_mutate(sequence, seq_length, dims):
    import numpy as np
    num_mutations = (seq_length * dims)*((seq_length - 1) * dims)
    mutations_matrix = np.zeros((seq_length,seq_length, dims*dims, seq_length,1,dims)) 

    for position1 in range(seq_length):
        
        for position2 in range(seq_length):
            
            for nuc1 in range(dims):
                
                for nuc2 in range(dims):
                    
                    mut_seq = np.copy(sequence)
                    mut_seq[0, position1, 0, :] = np.zeros(dims)
                    mut_seq[0, position1, 0, nuc1] = 1.0
                    mut_seq[0, position2, 0, :] = np.zeros(dims)
                    mut_seq[0, position2, 0, nuc2] = 1.0

                    mutations_matrix[position1, position2, (nuc1*dims)+nuc2, :] = mut_seq

    return mutations_matrix


        #Define a function that will take a sequence and output a saliency map generated by a first order mutagenesis

def firstordermut_saliency(X, layer, alphabet, figsize=(15,5)):

    ''' requires that deepomics is being used and the appropriate architecture has already been constructed
    Must first initialize the session and set best parameters

    layer is the activation layer we want to use as a string
    figsize is the figure size we want to use'''

    #first mutate the sequence
    X_mut = mutate(X, X.shape[1], X.shape[3])

    #take all the mutations and assign them into a dict for deepomics
    mutations = {'inputs': X_mut, 'targets': np.ones((X_mut.shape[0], 1))}
    #Get output or logits activations for the mutations
    mut_predictions = nntrainer.get_activations(sess, mutations, layer=layer)

    #take the WT and put it into a dict for deepomics
    WT = {'inputs': X, 'targets': np.ones((X.shape[0], 1))}
    #Get output or logits activations for the WT sequence
    predictions = nntrainer.get_activations(sess, WT, layer=layer)

    #shape the predictions of the mutations into the shape of a heatmap
    heat_mut = mut_predictions.reshape(X.shape[1],4).T
    
    #normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
    norm_heat_mut = heat_mut - predictions[0]
    norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

    plt.figure(figsize=figsize)
    visualize.plot_seq_pos_saliency(np.squeeze(X).T, 
                                        norm_heat_mut,
                                        alphabet=alphabet, 
                                        nt_width=400) 








































