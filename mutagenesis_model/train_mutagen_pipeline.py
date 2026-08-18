import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  Ridge
from sklearn.metrics import r2_score
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'motif_model'))
from motif_model import MotifModel
from model_utils import boot_strap
from data_preps import Motifs

# Input parameters from command line

n_mots = 647
bin_len = 700 #700 or 100
pool_f = 'avg' #'avg' or 'max'
act_f = 'relu' #'relu' or 'sigmoid'
strand_agg_ = 'sum'
act_thr = 4 # np.arange(-2, 18, 2)
motif_agg = 'avg' #'avg' or 'max'
alpha = 10**(-1.5)

alpha_exp = np.log10(alpha) # [0] +  np.logspace(-1, 6, 15, base=10).tolist()
alpha_str = f"10e{str(alpha_exp).replace('.', 'p')}"
strand_agg = None if strand_agg_ == "S1" else "sum"

S = np.copy(strand_agg_)
#########################
# Load motifs, conversion, and data. Prepare the data and get encoded sequences. 
with np.load('../input_files/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_647mots_LNCaP_aligned.npz') as data:
   motif_pwms = {key: data[key] for key in data.files}


with np.load('../input_files/HOCO_uniprot_to_gene.npz') as data:
    conversion = {key: data[key] for key in data.files}   


motifs = Motifs(motif_pwms, conversion)
mot_tfs = motifs.mot_tfs
tfs = np.unique(mot_tfs)


df_mut_all = pd.read_csv('../input_files/Complete_Mut_Data_LibVariant30_Pscount5.csv')

df_mut = df_mut_all[df_mut_all['#_Plasmids']>2]
df_mut = df_mut[df_mut['Variants'].str.split(' ').apply(len) < 3]

Y = df_mut['LFC2WT_MLE'].values*np.log(2)
mutated_seqs = df_mut['Mutated_seq'].values
wt_seqs = df_mut['WT_seq'].values


n_seqs, len_seq = len(mutated_seqs), len(mutated_seqs[0])
n_tfs, bin_len = len(tfs), len_seq
n_bins = int(len_seq / bin_len)

model = MotifModel(motifs, len_seq = len_seq)
X_mut = model.transform_seqs_to_bs( mutated_seqs, pool_f, bin_len, act_thr, act_f, strand_agg, motif_agg)
X_wt = model.transform_seqs_to_bs( wt_seqs, pool_f, bin_len, act_thr, act_f, strand_agg, motif_agg)

dX = X_mut - X_wt

regions = df_mut['Region'].values
unique_regions = np.unique(regions)

train_val_rgns, test_rgns = train_test_split(unique_regions, test_size=0.1, random_state=42, shuffle=True)

#########################

train_idx = np.where(np.isin(regions, train_val_rgns))[0]
test_idx = np.where(np.isin(regions, test_rgns))[0]

result = np.zeros(3)

X_train, X_test = dX[train_idx], dX[test_idx]
y_train, y_test = Y[train_idx], Y[test_idx]

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

joblib.dump(model, f'../output_files/Ridge_MutModel_BestConfig_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.joblib')
np.save(f'../output_files/Ridge_Results_MutModel_TestSet_BestConfig_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', result)
print(f"\nSuccess! All results saved to 'Ridge_Results_MutModel_TestSet_BestConfig_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.npy'")


coefs_bs = boot_strap(model, n_bootstraps=1000, X_train=X_train, y_train=y_train)
np.save(f'../output_files/Ridge_coef_bootstrapped_MutModel_BestConfig_bin{bin_len}bp_{pool_f}pooling_{S}_{act_f}thr{act_thr}_motagg{motif_agg}_alpha{alpha_str}_{n_mots}mots.npy', coefs_bs)
print(f"\nSuccess! All bootstrapped coefficients saved to 'Ridge_coef_bootstrapped_MutModel_BestConfig_bin{bin_len}bp_{pool_f}pooling_{act_f}_{S}.npy'")
