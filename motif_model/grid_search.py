from motif_model import MotifModel
from data_preps import TileDataset, Motifs
import numpy as np
import pandas as pd
from scipy import stats
import itertools
import pickle
import sys
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score


# Input parameters from command line

bin_len = int(sys.argv[1]) #700 or 100
pool_f = sys.argv[2] #'avg' or 'max'
act_f = sys.argv[3] #'relu' or 'sigmoid'
strand_agg_ = sys.argv[4] #'sum' or 'max' or None

strand_agg = None if strand_agg_ == "None" else strand_agg_

#######################
# Define the grid of parameters to search over.

alpha_list = [0] + np.logspace(-1, 6, 15, base=10).tolist()
activation_thrs = [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16]
motif_agg_methods = ['avg', 'max']
pool_functions = [pool_f] # ['avg', 'max']
activation_functions = [act_f] #, 'sigmoid']
strand_agg_methods = [strand_agg] #None or max or sum
n_strands, S = 1, 'S0'

if strand_agg_methods[0] is None:
    n_strands, S = 2, 'S1'


configs = []
for act_thr, pool_func, act_func, strand_agg, motif_agg in itertools.product(activation_thrs, pool_functions, activation_functions, strand_agg_methods, motif_agg_methods):

    configs.append({'act_thr': act_thr, 'pool_func': pool_func, 'act_func': act_func, 'strand_agg': strand_agg, 'motif_agg': motif_agg})

print("Total configurations to test:", len(configs))

#########################
# Load motifs, conversion, and data. Prepare the data and get encoded sequences. 

with np.load('../input_files/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_647mots_LNCaP_aligned.npz') as data:
    motif_pwms = {key: data[key] for key in data.files}

with np.load('../input_files/HOCO_uniprot_to_gene.npz') as data:
    conversion = {key: data[key] for key in data.files}

final_df = pd.read_csv('../input_files/Final_df_40excl_balanced.csv', index_col=0)


motifs = Motifs(motif_pwms, conversion)
mot_tfs = motifs.mot_tfs
tfs = np.unique(mot_tfs)

data = TileDataset(final_df)
X, X_RC = data.get_encoded_seqs()

len_seq = data.len_seq 
n_seqs, n_tfs = len(X), len(tfs)
n_bins = int(len_seq / bin_len)


test_idx = data.get_test_idx(test_frac = 0.1)
folds = data.generate_kfold_indx(kfolds = 5, test_frac = 0.1)


#########################
# Create storage for all results: A dictionary where each key is a config tuple and the value is an array of shape (n_seqs, n_tfs*n_bins*n_strands)

results = { (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg']): np.zeros((n_seqs,  n_tfs*n_bins*n_strands), dtype = np.float32) for c in configs }

batch_size = 100

model = MotifModel(motifs, len_seq = 700)

for i in range(0, n_seqs, batch_size):
    print(i)
    # 1. Get Raw Scores for this batch ONLY
    # model_t is the model we built in the previous step
    batch_f, batch_r = model.motif_conv_model().predict([X[i:i+batch_size], X_RC[i:i+batch_size]], verbose=0)

    # 2. Grid search over the RAW SCORES
    for c in configs:
        # Process Forward Strand
        x_f = model.shifted_activation(batch_f, threshold=c['act_thr'], method=c['act_func'])
        
        # Process Reverse Strand (Already flipped in the model_t we built!)
        x_r = model.shifted_activation(batch_r, threshold=c['act_thr'], method=c['act_func'])
        
        x_pooled_f = model.pooling(x_f, c['pool_func'], bin_len)
        x_pooled_r = model.pooling(x_r, c['pool_func'], bin_len)

        x_final = model.agg_mots_strands( x_pooled_f.numpy(), x_pooled_r.numpy(), strand_agg = c['strand_agg'], motif_agg = c['motif_agg'])
        # Store result
        key = (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg'])
        results[key][i:i+batch_size] = x_final


# Simple and preserves your tuple keys
with open(f'../output_files/FeaturesDict_gridsearch_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.pkl', 'wb') as f:
    pickle.dump(results, f)

########################

# Now we have all the features for all configs in the `results` dictionary. Next, we will loop through each config, train Ridge regression models for each target (lfc, dht, etoh) and each alpha, and store the metrics in a structured way.

master_results = {}

for c in configs:
    # 1. Create the key
    key = (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg'])
    
    # Initialize storage for this specific config
    # We use (3 metrics, 5 folds, len(alpha_list))
    config_data = {
        'lfc':  np.zeros((3, len(folds), len(alpha_list))),
        'dht':  np.zeros((3, len(folds), len(alpha_list))),
        'etoh': np.zeros((3, len(folds), len(alpha_list)))
    }

    for f_idx, fold in enumerate(folds):
        train_idx, val_idx = fold[0], fold[1]
        X_train, X_val = results[key][train_idx], results[key][val_idx]
        
        # Targets mapping
        y_train_map = {'lfc': data.y_lfc[train_idx], 'dht': data.y_dht[train_idx], 'etoh': data.y_etoh[train_idx]}
        y_val_map   = {'lfc': data.y_lfc[val_idx],   'dht': data.y_dht[val_idx],   'etoh': data.y_etoh[val_idx]}

        for a_idx, alpha in enumerate(alpha_list):
            for target in ['lfc', 'dht', 'etoh']:
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train_map[target])
                ypred = model.predict(X_val)

                # Calculate metrics
                mse = np.mean((ypred - y_val_map[target])**2)
                pcc, _ = stats.pearsonr(ypred, y_val_map[target])
                r2 = r2_score(y_val_map[target], ypred)

                # Store: 0=MSE, 1=PCC, 2=R2
                config_data[target][0, f_idx, a_idx] = mse
                config_data[target][1, f_idx, a_idx] = pcc
                config_data[target][2, f_idx, a_idx] = r2

    # Attach this config's data to the master dictionary
    master_results[key] = config_data
    print(f"Finished processing config: {key}")

# 2. Save everything into one single file
with open(f'../output_files/RidgeCV_3outputs_gridsearch_results_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.pkl', 'wb') as f:
    pickle.dump({
        'results': master_results,
        'alphas': alpha_list,
        'metrics_order': ['MSE', 'PCC', 'R2']
    }, f)

print(f"\nSuccess! All results saved to 'RidgeCV_3outputs_results_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.pkl'")

