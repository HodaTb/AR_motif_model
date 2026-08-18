import numpy as np
import random
from model_utils import get_encoded_seqs

class TileDataset:

    def __init__(self, df, *, len_seq = 700, pseudo_count = 5.0, random_seed = 42):
        self.df = df
        self.pseudo_count = pseudo_count
        self.len_seq = len_seq

        random.seed(random_seed)

        self._prepare_data()


    def _prepare_data(self):
        self.dht = 3*self.df['Tile_dht'].values + self.pseudo_count
        self.etoh = 3*self.df['Tile_etoh'].values + self.pseudo_count
        self.lib = self.df['lib'].values + self.pseudo_count
        self.seqs = self.df['Sequence'].values
        self.chrms = self.df['Chromosome'].values

    def get_encoded_seqs(self):

        return get_encoded_seqs(self.seqs, self.len_seq)

    @property
    def y_lfc(self):

        return np.log(self.dht/self.etoh)

    @property
    def y_dht(self):

        return np.log(self.dht/self.lib)
    
    @property
    def y_etoh(self):

        return np.log(self.etoh/self.lib)


    def _split_trainval_test(self, test_frac=0.10):

        fixed_seed = 14
        chroms = np.unique(self.chrms)

        rng = np.random.RandomState(fixed_seed)
        rng.shuffle(chroms)

        n = len(chroms)
        n_test = int(test_frac * n)

        chrom_test = chroms[:n_test]
        chrom_trainval = chroms[n_test:]

        test_idx = np.where(np.isin(self.chrms, chrom_test))[0]
        trainval_idx = np.where(np.isin(self.chrms, chrom_trainval))[0]

        return test_idx, chrom_trainval, chrom_test, trainval_idx


    def generate_kfold_indx(self, kfolds = 5, test_frac = 0.1):

        # Seed 14 + 11 chosen for chromosome balance

        _, chrom_trainval, _, _ = self._split_trainval_test(test_frac=test_frac)

        rng = np.random.RandomState(11)
        
        # 1. Isolate the indices belonging to trainval once
        trainval_all_idx = np.where(np.isin(self.chrms, chrom_trainval))[0]
        # Filter the chr_array to just the relevant chromosomes to speed up isin
        trainval_chr_array = self.chrms[trainval_all_idx]

        chroms = chrom_trainval.copy()

        rng.shuffle(chroms)
        chrom_folds = np.array_split(chroms, kfolds)

        folds = []
        for chrom_group in chrom_folds:
            # 2. Find which indices in the TRAINVAL set belong to the VAL group
            is_val = np.isin(trainval_chr_array, chrom_group)
            
            # 3. Map back to original indices
            val_idx = trainval_all_idx[is_val]
            train_idx = trainval_all_idx[~is_val]

            folds.append((train_idx, val_idx))

        return folds    


    def get_test_idx(self, test_frac = 0.1):
         
         test_ind, _, _, _ = self._split_trainval_test(test_frac=test_frac)

         return test_ind

    def get_trainval_idx(self, test_frac = 0.1):
         
         _, _, _, trainval_ind = self._split_trainval_test(test_frac=test_frac)

         return trainval_ind





class Motifs:

    def __init__(self, motif_pwms, conversion):
        self.motif_pwms = motif_pwms
        self.conversion = conversion
        self.motif_names = list(self.motif_pwms.keys())
        self.len_mot_dict = self._build_len_dict()
        self.mot_names_inmodel = self._get_mot_names_inmodel()
        self.mot_tfs = self._motif_tfs()


    def _build_len_dict(self):

        len_mot_dict = {}
        for name in self.motif_names:
            lmot = len(self.motif_pwms[name])
            if lmot not in len_mot_dict:
                len_mot_dict[lmot] = []
            len_mot_dict[lmot].append(name)

        return len_mot_dict
    

    def _get_mot_names_inmodel(self):

        mot_names_inmodel = []
        for L in self.len_mot_dict:
            mot_names_inmodel += self.len_mot_dict[L]

        return np.array(mot_names_inmodel)
    

    def _motif_tfs(self):

        mot_tfs = []
        for i in self.mot_names_inmodel:
            if 'H12CORE' in i:

                a = i.split('.')[0]
                mot_tfs.append(self.conversion[a].tolist())

            else:
                mot_tfs.append(i.split('.')[-1])

        return np.array(mot_tfs)
