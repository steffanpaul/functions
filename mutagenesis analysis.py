## All the plotting functions I use when developing mutagenesis tools


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

''' PETER'S BACKPROP SALIENCY VISUALIZATION ''' 

# sort activations
plot_index = np.argsort(predictionsoutput[:,0])[::-1]

# saliency parameters  <-- neccessary because we need to rebuild graph using different gradient ops
params = {'genome_model': cnn_model, 
          'input_shape': input_shape, 
          'output_shape': output_shape, 
          'optimization': optimization,
          'model_path': param_path+'_best.ckpt',
         }
# plot saliency of top activations
num_plots = 5
for index in plot_index[:num_plots]:

        # get a test sequence
        X = np.expand_dims(test['inputs'][index], axis=0)

        # guided backprop
        guided_saliency = saliency.guided_backprop(X, layer='output', class_index=None, params=params)

        # plot and save saliency
        fig = plt.figure(figsize=(15,3))
        visualize.plot_seq_pos_saliency(np.squeeze(X).T, 
                                        np.squeeze(guided_saliency[0]).T, 
                                        alphabet='rna', 
                                        nt_width=400)




''' FIRST ORDER MUTAGENESIS SALIENCY ANALYSIS '''

    ''' requires that deepomics is being used and the appropriate architecture has already been constructed
    Must first initialize the session and set best parameters

    layer is the activation layer we want to use as a string
    figsize is the figure size we want to use'''

# sort activations
plot_index = np.argsort(predictionsoutput[:,0])[::-1]

X = np.expand_dims(test['inputs'][plot_index[0]], axis=0)

X_mut = mf.mutate(X, X.shape[1], X.shape[3])
mutations = {'inputs': X_mut, 'targets': Y_test[:X_mut.shape[0]]}

# initialize session
sess = utils.initialize_session()

# set best parameters
nntrainer.set_best_parameters(sess)

#Get output activations for the mutations
mut_predictionslogits = nntrainer.get_activations(sess, mutations, layer='dense_1_bias')
mut_predictionsoutput= nntrainer.get_activations(sess, mutations, layer='output')

#shape the predictions of the mutations into the shape of a heatmap
heat_mut = mut_predictionslogits.reshape(X.shape[1],4).T

#normalize the heat map rearrangement by minusing it by the true prediction score of that test sequence
norm_heat_mut = heat_mut - predictionslogits[plot_index[0]]
norm_heat_mut = utils.normalize_pwm(norm_heat_mut, factor=4)

plt.figure(figsize=(15,3))
visualize.plot_seq_pos_saliency(np.squeeze(X).T, 
                                        norm_heat_mut, 
                                        alphabet='rna', 
                                        nt_width=400)