"""Trains the best-config Ridge model (from grid_search.py's results) for one
response variable, then computes its mutational and TF contribution profiles."""
from data_preps import Motifs, TileDataset
from model_utils import boot_strap
from mutational_profiles import MutationalProfiles, ContributionProfiles
from motif_model import MotifModel
import numpy as np
import pandas as pd
from scipy import stats
import pickle
import sys
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score
import joblib
from functools import partial
import json


model_name = sys.argv[1]
n_mots = int(sys.argv[2])

with open(f"../input_files/best_models_hyperparams_MSE_{n_mots}mots.json") as f:
    best_configs = json.load(f)

config = best_configs[model_name]

print("Running:", model_name)
print(config)


# Input parameters from command line

bin_len = int(model_name.split('bp')[0]) #700 or 100
pool_f = config["pooling"] #'avg' or 'max'
pool_f = str.lower(pool_f).split('pooling')[0]
act_f = config["activation_function"] #'relu' or 'sigmoid'
act_f = str.lower(act_f)
strand_agg_ = model_name.split('_')[1] #S0 or S1
act_thr = float(config["activaition_threshold"]) # np.arange(-2, 18, 2)
motif_agg = config["motif_agg_method"] #'avg' or 'max'
motif_agg = 'avg' if motif_agg == 'Motifs_mean' else 'max'
alpha = float(config["alpha_value"])
output =  model_name.split('_')[2]  #'lfc' or 'dht' or 'etoh'

alpha_exp = np.log10(alpha) # [0] +  np.logspace(-1, 6, 15, base=10).tolist()
alpha_str = f"10e{str(alpha_exp).replace('.', 'p')}"
strand_agg = None if strand_agg_ == "S1" else "sum"

S = np.copy(strand_agg_)
#########################
# Load motifs, conversion, and data. Prepare the data and get encoded sequences. 


final_df = pd.read_csv('../input_files/Final_df_40excl_balanced.csv', index_col=0)

data = TileDataset(final_df)

test_idx = data.get_test_idx(test_frac = 0.1)
train_idx = np.setdiff1d(np.arange(len(final_df)), test_idx)

#########################

# Load the pre-computed feature dict from grid_search.py
with open(f'../output_files/FeaturesDict_gridsearch_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.pkl', 'rb') as f:
    results = pickle.load(f)


########################
key = (act_thr, pool_f, act_f, strand_agg, motif_agg)

result = np.zeros(3)

X_train, X_test = results[key][train_idx], results[key][test_idx]
y_train, y_test = getattr(data, f'y_{output}')[train_idx], getattr(data, f'y_{output}')[test_idx] 

model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
ypred = model.predict(X_test)

# Calculate metrics
mse = np.mean((ypred - y_test)**2)
pcc, _ = stats.pearsonr(ypred, y_test)
r2 = r2_score(y_test, ypred)

# Store: 0=MSE, 1=PCC, 2=R2
result[0] = mse
result[1] = pcc
result[2] = r2

# Attach this config's data to the master dictionary
print(f"Finished processing config: {key}")

joblib.dump(model, f'../output_files/Ridge_model_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.joblib')
np.save(f'../output_files/Ridge_results_test_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', result)

print(f"\nSuccess! Results saved to 'Ridge_results_test_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy'")

coefs_bs = boot_strap(model, n_bootstraps=1000, X_train=X_train, y_train=y_train)

np.save(f'../output_files/Ridge_coef_bootstrapped_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', coefs_bs)

print(f"\nSuccess! Bootstrapped coefficients saved to 'Ridge_coef_bootstrapped_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy'")

##############################
# Caluclate mutational profiles for the test set sequences using the trained model and the same feature transformation pipeline.

with np.load(f'../input_files/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_{n_mots}mots_LNCaP_aligned.npz') as data:
    motif_pwms = {key: data[key] for key in data.files}

with np.load('../input_files/HOCO_uniprot_to_gene.npz') as data:
    conversion = {key: data[key] for key in data.files}   

mut_seqs = pd.read_csv('../input_files/mutagenesis_region_reference.csv')['WT_seq'].values

motifs_nmots = Motifs(motif_pwms, conversion)
model_nmots = MotifModel(motifs_nmots, len_seq = 700)

transformer_func = partial(model_nmots.transform_seqs_to_bs, 
                           pool_type=pool_f, bin_len=bin_len, act_thr=act_thr, 
                           act_func=act_f, strand_agg=strand_agg, motif_agg=motif_agg)


mut_calculator = MutationalProfiles(model, transformer_func)
mut_profs = mut_calculator.compute_mutational_profiles(mut_seqs)
np.save(f'../output_files/Mutational_profiles_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', mut_profs)

print(f"\nSuccess! Mutational profiles saved to 'Mutational_profiles_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy'")


# Now calculate contribution profiles for each TF by multiplying the mutational profiles with the model coefficients. This will give us a more interpretable view of how each TF contributes to the predictions across the sequence.
cont_calculator = ContributionProfiles(model, transformer_func)
n_tfs = len(np.unique(motifs_nmots.mot_tfs))
tf_indices = [[i] for i in range(n_tfs)]

# 4. Calculate the scores
# This will return an array of shape (n_tfs, n_seqs, 309, 4)
scores_tfs = cont_calculator.compute_contribution_profiles(mut_seqs, tf_indices)

np.save(f'../output_files/Contribution_profiles_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', scores_tfs)

print(f"\nSuccess! Contribution profiles saved to 'Contribution_profiles_{output}_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy'")