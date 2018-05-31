## deepomics basic architecture for training


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
import tensorflow as tf
from deepomics import neuralnetwork as nn
from deepomics import utils, fit, visualize, saliency

-------------------------------------------------------------------------------------------------------

#example architecture

def cnn_model(input_shape, output_shape):
    
    layer1 = {'layer': 'input', #200          
              'input_shape': input_shape
             }
    layer2 = {'layer': 'conv1d',        # input, conv1d, dense, conv1d_residual, dense_residual, conv1d_transpose,
                                        # concat, embedding, variational_normal, variational_softmax, + more
              'num_filters': 20,
              'filter_size': 15,
              'norm': 'batch',          # if removed, automatically adds bias instead
              'activation': 'relu',     # or leaky_relu, prelu, sigmoid, tanh, etc
              'dropout': 0.2,           # if removed, default is no dropout
              'padding': 'SAME',        # VALID OR SAME
              'max_pool': 20,           # 
             }
    layer3 = {'layer': 'dense',
              'num_units': 64,
              'norm': 'batch',
              'activation': 'relu',
              'dropout': 0.5,
             }
    layer4 = {'layer': 'dense',
              'num_units': output_shape[1],
              'activation': 'sigmoid'
             }

    model_layers = [layer1, layer2, layer3, layer4]

    # optimization parameters
    optimization = {"objective": "binary",     # binary, categorical, squared_error, elbo
                    "optimizer": "adam",       # adam, sgd, momentum, rmsprop, adagrad
                    "learning_rate": 0.0003,  
                    "l2": 1e-6,
                    #"label_smoothing": 0.05,
                    #"l1": 1e-6,
                    }
    return model_layers, optimization

-------------------------------------------------------------------------------------------------------

tf.reset_default_graph()

# get shapes of inputs and targets
input_shape = list(train['inputs'].shape)
input_shape[0] = None
output_shape = train['targets'].shape

# load model parameters
model_layers, optimization = cnn_model(input_shape, output_shape)

# build neural network class
nnmodel = nn.NeuralNet(seed=247)
nnmodel.build_layers(model_layers, optimization)

# compile neural trainer
save_path = '../results/CTCF'
param_path = os.path.join(save_path, 'deepomics_cnn')
nntrainer = nn.NeuralTrainer(nnmodel, save='best', file_path=param_path)


-------------------------------------------------------------------------------------------------------


# look at the shapes of each layer
nnmodel.inspect_layers()


-------------------------------------------------------------------------------------------------------

# initialize session
sess = utils.initialize_session()


-------------------------------------------------------------------------------------------------------

#Train the model

# fit model
data = {'train': train, 'valid': valid}
fit.train_minibatch(sess, nntrainer, data, 
                    batch_size=100, 
                    num_epochs=100,
                    patience=10, 
                    verbose=2, 
                    shuffle=True, 
                    save_all=False)

-------------------------------------------------------------------------------------------------------

# set best parameters
nntrainer.set_best_parameters(sess)

# test model
loss, mean_vals, std_vals = nntrainer.test_model(sess, test, name='test')



























