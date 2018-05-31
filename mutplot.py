## All the plotting functions I use when developing mutagenesis tools


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Peter's backprop saliency visualizing

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