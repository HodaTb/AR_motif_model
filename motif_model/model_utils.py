import random
import numpy as np
from sklearn.utils import resample
from dna_utils import RC, pad_to_Nbp, generate_random_seq



def get_encoded_seqs(seqs, len_seq):

    X = []
    X_RC = []

    random_seed = 42
    random.seed(random_seed)

    base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    random_seq = generate_random_seq(k = len_seq, probabilities = {'A': 0.3, 'T': 0.3, 'C': 0.2, 'G': 0.2})

    for seq in seqs:

        padded_seq = pad_to_Nbp(seq, random_seq,  len_seq).upper()
        
        encoded = [base_map.get(char, 0) for char in padded_seq]
        encoded_rc = [base_map.get(char, 0) for char in RC(padded_seq)]
        
        X.append(encoded)
        X_RC.append(encoded_rc)

    return np.array(X), np.array(X_RC)



def boot_strap(model, n_bootstraps, X_train, y_train, random_state=42):
    coefficients = []
    for i in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state + i)
        model.fit(X_resampled, y_resampled)
        coefficients.append(model.coef_)
        print(i)

    return np.array(coefficients)
