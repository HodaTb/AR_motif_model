"""In-silico saturation mutagenesis: per-position/allele predicted effects
(MutationalProfiles) and their per-TF decomposition (ContributionProfiles)."""
import numpy as np

class MutationalProfiles:
    
    def __init__(self,  model, transformer):
        self.model = model
        self.transformer = transformer
    
    def mutate_seqs(self, seqs):

        all_mutated = []

        for k in range(len(seqs)):
            seq = list(seqs[k])

            for i in range(len(seq)):
                m1 = seq.copy()
                m2 = seq.copy()
                m3 = seq.copy()
                m4 = seq.copy()


                m1[i] = 'A'
                m2[i] = 'C'
                m3[i] = 'G'
                m4[i] = 'T'


                all_mutated.append("".join(m1))
                all_mutated.append("".join(m2))
                all_mutated.append("".join(m3))
                all_mutated.append("".join(m4))

        all_mutated = np.array(all_mutated)

        return all_mutated



    def compute_mutational_profiles(self, seqs):

        seq_len = len(seqs[0])
        n_seqs = len(seqs)

        Xww = self.transformer(seqs)

        scores = []

        for s in range(len(seqs)):

            mutated_all = self.mutate_seqs([seqs[s]])
            Xmm = self.transformer(mutated_all)

            pw_ = self.model.predict(np.reshape(Xww[s], (-1,len(Xww[s]))))
            pm_ = self.model.predict(Xmm)
                
            pm_ = pm_.reshape((1,seq_len,4))
            pw_ = pw_[:, np.newaxis, np.newaxis] 

            score = -(pm_-pw_)

            scores.append(score)

        scores = np.reshape(scores, (n_seqs, seq_len, 4))
        return scores  
    

class ContributionProfiles(MutationalProfiles):
    """
    Calculates importance scores by detangling individual TF contributions 
    using model coefficients (coef_ and intercept_).
    """
    def compute_contribution_profiles(self, seqs, TF_inds, batch_pos=100):
        seq_len = len(seqs[0])
        n_seqs = len(seqs)
        n_clusters = len(TF_inds)

        # Transform wild-type sequences
        Xww = self.transformer(seqs)

        # Initialize scores array: (n_tfs, n_seqs, seq_len, 4)
        all_scores = [[] for _ in range(n_clusters)]

        for s in range(n_seqs):
            mutated_all = self.mutate_seqs([seqs[s]])

            # Transform mutants in position-batches (mutate_seqs orders them 4 per
            # position: A,C,G,T) instead of one oversized predict() call on all
            # 4*seq_len mutants at once -- same batching compute_profile_batched_700
            # uses, needed here too now that seqs can be full 700bp regions.
            Xmm_parts = [self.transformer(mutated_all[bs:bs + batch_pos * 4])
                         for bs in range(0, len(mutated_all), batch_pos * 4)]
            Xmm = np.concatenate(Xmm_parts, axis=0)

            for c, indices in enumerate(TF_inds):
                # Calculate weighted wild-type: dot product of features and coefficients
                # Resulting shape: (1,)
                pw_ = np.dot(Xww[s, indices], self.model.coef_[indices]) + self.model.intercept_
                
                # Calculate weighted mutant: dot product for all 4*seq_len mutants
                # Resulting shape: (seq_len * 4,)
                pm_ = np.dot(Xmm[:, indices], self.model.coef_[indices]) + self.model.intercept_
                
                # Reshape for broadcasted subtraction
                pm_reshaped = pm_.reshape((1, seq_len, 4))
                pw_reshaped = pw_  # Scalar-like for the subtraction

                score = -(pm_reshaped - pw_reshaped)
                all_scores[c].append(score)

        return np.array(all_scores).reshape((n_clusters, n_seqs, seq_len, 4))