"""Grid search over pooling/activation/motif-aggregation hyperparameters for the
model fit directly to the saturation-mutagenesis dataset."""
# %%
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import itertools
import sys
import os
import pickle
from scipy import stats
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'motif_model'))
from model_utils import get_encoded_seqs
from data_preps import Motifs
from motif_model import MotifModel

df_mut_all = pd.read_csv('../input_files/Complete_Mut_Data_LibVariant30_Pscount5.csv')

df_mut = df_mut_all[df_mut_all['#_Plasmids']>2]
df_mut = df_mut[df_mut['Variants'].str.split(' ').apply(len) < 3]

Y = df_mut['LFC2WT_MLE'].values*np.log(2)
mutated_seqs = df_mut['Mutated_seq'].values
wt_seqs = df_mut['WT_seq'].values

X_mut, XRC_mut = get_encoded_seqs(mutated_seqs, len_seq= 309)
X_wt, XRC_wt = get_encoded_seqs(wt_seqs, len_seq = 309)

act_f = sys.argv[1] #'relu' or 'sigmoid'

#######################
# Define the grid of parameters to search over.

alpha_list = [0] + np.logspace(-2, 4, 13, base = 10).tolist()
activation_thrs = [-2, 0,  2, 4, 6, 8, 10, 12, 14, 16]
motif_agg_methods = ['avg', 'max']
pool_functions = ['avg', 'max']
activation_functions = [act_f] #['relu', 'sigmoid']
strand_agg_ = 'sum' #'sum' or 'max' or None
strand_agg = None if strand_agg_ == "None" else strand_agg_
strand_agg_methods = [strand_agg] #None or max or sum
n_strands, S = 1, 'S0'

if strand_agg_methods[0] is None:
    n_strands, S = 2, 'S1'


configs = []
for act_thr, pool_func, act_func, strand_agg, motif_agg in itertools.product(activation_thrs, pool_functions, activation_functions, strand_agg_methods, motif_agg_methods):

    configs.append({'act_thr': act_thr, 'pool_func': pool_func, 'act_func': act_func, 'strand_agg': strand_agg, 'motif_agg': motif_agg})

print(configs)
print("Total configurations to test:", len(configs))

#########################
# Load motifs, conversion, and data. Prepare the data and get encoded sequences. 

with np.load('../input_files/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_647mots_LNCaP_aligned.npz') as data:
   motif_pwms = {key: data[key] for key in data.files}


with np.load('../input_files/HOCO_uniprot_to_gene.npz') as data:
    conversion = {key: data[key] for key in data.files}   


motifs = Motifs(motif_pwms, conversion)
mot_tfs = motifs.mot_tfs
tfs = np.unique(mot_tfs)


n_seqs, len_seq = np.shape(X_mut)[0], np.shape(X_mut)[1]
n_tfs, bin_len = len(tfs), len_seq
n_bins = int(len_seq / bin_len)


#########################
# Create storage for all results: A dictionary where each key is a config tuple and the value is an array of shape (n_seqs, n_tfs*n_bins*n_strands)

results = { (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg']): np.zeros((n_seqs,  n_tfs*n_bins*n_strands), dtype = np.float32) for c in configs }

batch_size = 100

model = MotifModel(motifs, len_seq = len_seq)

for i in range(0, n_seqs, batch_size):
    print(i)
    # 1. Get Raw Scores for this batch ONLY
    # model_t is the model we built in the previous step
    batch_f_mut, batch_r_mut = model.motif_conv_model().predict([X_mut[i:i+batch_size], XRC_mut[i:i+batch_size]], verbose=0)
    batch_f_wt, batch_r_wt = model.motif_conv_model().predict([X_wt[i:i+batch_size], XRC_wt[i:i+batch_size]], verbose=0)

    # 2. Grid search over the RAW SCORES
    for c in configs:
        # Process Forward Strand
        x_f_mut = model.shifted_activation(batch_f_mut, threshold=c['act_thr'], method=c['act_func'])
        x_f_wt = model.shifted_activation(batch_f_wt, threshold=c['act_thr'], method=c['act_func'])

        # Process Reverse Strand (Already flipped in the motif_conv_model() we built!)
        x_r_mut = model.shifted_activation(batch_r_mut, threshold=c['act_thr'], method=c['act_func'])
        x_r_wt = model.shifted_activation(batch_r_wt, threshold=c['act_thr'], method=c['act_func'])

        x_pooled_f_mut = model.pooling(x_f_mut, c['pool_func'], bin_len)
        x_pooled_r_mut = model.pooling(x_r_mut, c['pool_func'], bin_len)

        x_pooled_f_wt = model.pooling(x_f_wt, c['pool_func'], bin_len)
        x_pooled_r_wt = model.pooling(x_r_wt, c['pool_func'], bin_len)

        x_final_mut = model.agg_mots_strands( x_pooled_f_mut.numpy(), x_pooled_r_mut.numpy(), strand_agg = c['strand_agg'], motif_agg = c['motif_agg'])
        x_final_wt = model.agg_mots_strands( x_pooled_f_wt.numpy(), x_pooled_r_wt.numpy(), strand_agg = c['strand_agg'], motif_agg = c['motif_agg'])

        # Store result
        key = (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg'])
        results[key][i:i+batch_size] = x_final_mut - x_final_wt


# Simple and preserves your tuple keys
with open(f'../output_files/FeaturesDict_dXMut_gridsearch_{n_tfs}tfs_bin{bin_len}bp_{act_f}_{S}.pkl', 'wb') as f:
    pickle.dump(results, f)


#################
#kfold on regions

regions = df_mut['Region'].values
unique_regions = np.unique(regions)

train_val_rgns, test_rgns = train_test_split(unique_regions, test_size=0.1, random_state=42, shuffle=True)
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
train_rgns = []
val_rgns = []
# Use the split method to generate indices to split data into training and test set
for train_index, val_index in kf.split(train_val_rgns):    
    train_rgns.append(train_val_rgns[train_index])
    val_rgns.append(train_val_rgns[val_index])

##################


master_results = {}

for c in configs:
    # 1. Create the key
    key = (c['act_thr'], c['pool_func'], c['act_func'], c['strand_agg'], c['motif_agg'])

    config_data = np.zeros((3, k_folds, len(alpha_list)))

    for fold in range(5):
        train_idx = np.where(np.isin(regions, train_rgns[fold]))[0]
        val_idx = np.where(np.isin(regions, val_rgns[fold]))[0]

        X_train, X_val = results[key][train_idx], results[key][val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]

        for a_idx, alpha in enumerate(alpha_list):
            model = Ridge(alpha= alpha)

            model.fit(X_train, y_train)
            ypred = model.predict(X_val)

            mse = np.sum((ypred - y_val)**2) / len(y_val)
            pcc, _ = stats.pearsonr(ypred,y_val)
            r2 = r2_score(y_val, ypred)

            config_data[0, fold, a_idx] = mse
            config_data[1, fold, a_idx] = pcc
            config_data[2, fold, a_idx] = r2

    master_results[key] = config_data
    print(f"Finished processing config: {key}")

# 2. Save everything into one single file
with open(f'../output_files/RidgeCV_MutModel_gridsearch_{n_tfs}tfs_results_bin{bin_len}bp_{act_f}_{S}.pkl', 'wb') as f:
    pickle.dump({
        'results': master_results,
        'alphas': alpha_list,
        'metrics_order': ['MSE', 'PCC', 'R2']
    }, f)

print(f"\nSuccess! Results saved to 'RidgeCV_MutModel_gridsearch_{n_tfs}tfs_results_bin{bin_len}bp_{act_f}_{S}.pkl'")


