import numpy as np
import random

def generate_random_seq(k = 1000, probabilities = {'A': 0.3, 'T': 0.3, 'C': 0.2, 'G': 0.2}):

    random.seed(42)
    random_seq = random.choices(
        population=list(probabilities.keys()), 
        weights=list(probabilities.values()), 
        k= k
    )
    return ''.join(random_seq)


def pad_to_Nbp(dna_seq, random_seq, N):

    total_length = N
    current_length = len(dna_seq)
    padding_length = total_length - current_length
 
    # Concatenating the random sequence to the original sequence

    return dna_seq + random_seq[:padding_length]


def RC(seq):

    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N':'N'}
    seq_rc = "".join(complement.get(base, base) for base in reversed(seq))
    return seq_rc
