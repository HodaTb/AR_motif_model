import tensorflow as tf
import numpy as np
import keras
from model_utils import get_encoded_seqs

#Feb 2026
#Hoda Taeb
# Legacy model definition using the TF/Keras-2 (tf.keras) API (tf.one_hot instead of keras.ops.one_hot).
# Used by gwas_variant_analysis/fig6_gwas_variants.ipynb, which runs the model's forward
# pass directly (scoring new GWAS variant sequences) rather than only loading cached
# results -- requires an environment with Keras 2 (not Keras 3, unlike the rest of the
# repo). All other notebooks use motif_model.py (Keras 3) for current work.

class MotifModel:

    def __init__(self, motif_obj, *, len_seq = 700):
        self.motifs = motif_obj
        self.motif_pwms = self.motifs.motif_pwms
        self.len_mot_dict = self.motifs.len_mot_dict
        self.len_seq = len_seq
        self.mot_tfs = self.motifs.mot_tfs
        self.conv_model = self.motif_conv_model()


    def _motif_layer(self, mot_list_L):
        # mot_list_L is a list of motifs that all havce the same length
        # sigma is a cutoff that controls the size of the score cutoff
        
        mot_len = len(self.motif_pwms[mot_list_L[0]])  # length of motif
        num_mots = len(mot_list_L)            # number of motifs
        
        Efs = np.zeros(num_mots)  #updated to handle this log(pi/0.25) we need 0 biases
        ws = np.array([self.motif_pwms[mname].T for mname in mot_list_L]).T

        motif_conv_layer = tf.keras.layers.Conv1D(filters = num_mots, kernel_size = mot_len, padding='same', trainable=False, kernel_initializer=tf.keras.initializers.Constant(ws),
        bias_initializer=tf.keras.initializers.Constant(Efs))   # convolves seq with motifs of length mot_len
        
        return motif_conv_layer

    def motif_conv_model(self):

        # len_mot_dict = dictionary indexed by length L that has list of motifs with that L
        # len_seq = max size of input
        # sigma = score cutoff param for motifs
        # bin_len  = window for downsampling
        
        inputs = tf.keras.layers.Input(shape = (self.len_seq, ), dtype='int32')
        inputs_RC = tf.keras.layers.Input(shape = (self.len_seq, ), dtype='int32')     # RC input

        # hot encode sequences
        x_hot = tf.one_hot(inputs - 1, 4)
        x_hot_RC = tf.one_hot(inputs_RC - 1, 4)

            
        # apply motifs as convolutions
        outs = []
        outs_RC = []
        for L in self.len_mot_dict:    # iterate over all motif lengths

            E_layer = self._motif_layer(self.len_mot_dict[L])
            
            out = E_layer(x_hot)
            outs.append(out)

            out_RC = E_layer(x_hot_RC)
            out_RC_flipped = tf.keras.layers.Lambda(lambda x: x[:, ::-1, :])(out_RC)
            outs_RC.append(out_RC_flipped)

        # concatenate all motif output together
        combined_outs = tf.keras.layers.Concatenate(axis=-1)(outs)
        combined_outs_RC = tf.keras.layers.Concatenate(axis=-1)(outs_RC)

        conv_model = tf.keras.Model(inputs=[inputs, inputs_RC], outputs=[combined_outs, combined_outs_RC])

        return conv_model
    
    @staticmethod
    def shifted_activation(x, threshold=0.0, method='relu'):
        # Centering the thresholding logic
        x_shifted = x - tf.cast(threshold, x.dtype)
        if method == 'relu':
            return tf.nn.relu(x_shifted)
        elif method == 'sigmoid':
            return tf.nn.sigmoid(x_shifted)
        return x_shifted

    @staticmethod
    def pooling(x, pool_type, bin_len):
        if pool_type == 'avg':
            return tf.nn.avg_pool1d(x, ksize=bin_len, strides=bin_len, padding='VALID')
        else:
            return tf.nn.max_pool1d(x, ksize=bin_len, strides=bin_len, padding='VALID')
    

    
    def agg_mots_strands( self, X_mots, X_mots_RC, strand_agg = 'sum', motif_agg = 'avg'):
        """
        Aggregates motif hits by TF, handling strand and motif-level grouping.
        
        Parameters:
        -----------
        mot_tfs : ndarray
            1D array/list mapping each motif index to a TF name.
        X_mots : ndarray
            3D array (n_seqs, n_bins, n_motifs) for Forward strand.
        X_mots_RC : ndarray
            3D array (n_seqs, n_bins, n_motifs) for RC strand.
        strand_agg : str or None, default='sum'
            Options:
            - 'sum' : Add Forward and RC signals.
            - 'max' : Take element-wise maximum of strands.
            - None  : Keep strands separate (Block Forward then Block RC).
        motif_agg : str, default='avg'
            Options:
            - 'avg' : Mean of all motifs belonging to a TF.
            - 'max' : Maximum signal among motifs for a TF.

        Returns:
        --------
        ndarray : 2D array (n_seqs, n_features)

        """


        tf_unique = np.unique(self.mot_tfs)
        n_tfs = len(tf_unique)
        n_seqs, n_bins = np.shape(X_mots)[0], np.shape(X_mots)[1]

        mot_agg_func = np.mean if motif_agg == 'avg' else np.max

        if strand_agg is not None:
            X_tfs = np.zeros((n_seqs, n_bins, n_tfs))
        else:
            # Doubling the TF dimension to hold both strands [Forward, RC]
            X_tfs = np.zeros((n_seqs, n_bins, n_tfs * 2))
        

        if strand_agg == 'sum':
            X_S0 = X_mots + X_mots_RC

        elif strand_agg == 'max':
            X_S0 = np.maximum(X_mots, X_mots_RC)
        

        for i, TF in enumerate(tf_unique):
            motifs_mask = (self.mot_tfs == TF)
                
            if strand_agg is not None:

                X_tfs[:,:, i] = mot_agg_func(X_S0[:, :, motifs_mask], axis = -1)

            else:

                X_tfs[:, :, i] = mot_agg_func(X_mots[:, :, motifs_mask], axis = -1)
                X_tfs[:, :, i + n_tfs] = mot_agg_func(X_mots_RC[:, :, motifs_mask], axis = -1)
    

        if strand_agg is not None:
            # Returns (n_seqs, n_bins * n_tfs)
            return X_tfs.reshape(n_seqs, -1)
        
        else:
            # Match the "SS" logic: Concatenate [Flattened Forward] and [Flattened RC]
            # X_tfs[:, :, :n_tfs] is the Forward block
            # X_tfs[:, :, n_tfs:] is the RC block
            fw_flat = X_tfs[:, :, :n_tfs].reshape(n_seqs, -1)
            rc_flat = X_tfs[:, :, n_tfs:].reshape(n_seqs, -1)
            
            # Returns (n_seqs, 2 * n_bins * n_tfs)
            return np.concatenate((fw_flat, rc_flat), axis=1)       


    def transform_seqs_to_bs(self, seqs, pool_type, bin_len, act_thr, act_func, strand_agg, motif_agg):

        Xs, Xs_RC = get_encoded_seqs(seqs, self.len_seq)
        
        X_conv, X_conv_RC = self.conv_model.predict([Xs, Xs_RC], verbose = 0)

        X_act = self.shifted_activation(X_conv, threshold=act_thr, method=act_func)
        X_act_RC = self.shifted_activation(X_conv_RC, threshold=act_thr, method=act_func)

        X_pooled = self.pooling(X_act, pool_type=pool_type, bin_len=bin_len)
        X_pooled_RC = self.pooling(X_act_RC, pool_type=pool_type, bin_len=bin_len)

        # Now we have the raw motif scores for both strands. Next step is to aggregate them by TF and handle strand aggregation.

        X_final = self.agg_mots_strands(np.array(X_pooled), np.array(X_pooled_RC), strand_agg=strand_agg, motif_agg=motif_agg)

        return X_final
