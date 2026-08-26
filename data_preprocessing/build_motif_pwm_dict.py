"""Build the aligned, LNCaP-filtered PWM dictionary (Methods 2.3-2.4) from JASPAR
and HOCOMOCO position-frequency matrices.

Required raw inputs (place in ../raw_data/):
  - H12CORE_pcms.txt -- HOCOMOCO v12 CORE PCMs (JASPAR-format text), from
    https://hocomoco14.autosome.org/downloads_v12
  - JASPAR_2024_CORE_last_version_HomoSapiens.txt -- JASPAR 2024 CORE PFMs
    (Homo sapiens), from https://jaspar.elixir.no/
  - Expression_Public_25Q2_LNCaP.xlsx -- DepMap Public 25Q2 expression data,
    LNCaP row only, from https://depmap.org/portal/download/all/

Also required: ../input_files/HOCO_uniprot_to_gene.npz (already in this repo).

Output: ../input_files/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_647mots_LNCaP_aligned.npz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from scipy import stats

RAW_DIR = '../raw_data'
INPUT_DIR = '../input_files'
OUTPUT_PATH = f'{INPUT_DIR}/motifs_dict_PWM_pseudo_bg_exp3_JaspHomoLV24_293tf_647mots_LNCaP_aligned.npz'

# Visual QC (section 6) plots every TF-group's motifs after alignment; it does
# not affect the saved output. Off by default so this runs non-interactively.
RUN_VISUAL_QC = False


def calculate_information_content(pwm):
    """Information content in bits at each position (uniform 0.25 background)."""
    info_content = []
    for position in pwm:
        ic = 2 + np.sum(position * np.log2(position))
        info_content.append(ic)
    return np.array(info_content)


def logo(ppm, tf):
    """Plot a sequence logo for visual QC."""
    info_content = calculate_information_content(ppm)
    ppm_bits = ppm * info_content[:, np.newaxis]
    ppm_df = pd.DataFrame(ppm_bits, columns=['A', 'C', 'G', 'T'])
    logo_plot = logomaker.Logo(ppm_df)
    logo_plot.ax.figure.set_size_inches(len(ppm) * 0.7, 1)
    logo_plot.ax.set_title(f'{tf}')
    logo_plot.ax.set_xlabel('Position')
    logo_plot.ax.set_ylabel('bits')
    logo_plot.ax.set_ylim(0, 2)
    plt.show()


def parse_motifs(file_path):
    """Parse a JASPAR-format PFM text file into {motif_id: PFM} and a parallel
    array of TF names (the last '.'-separated token of each motif id)."""
    motifs_, name, matrix = {}, None, []
    tf_names = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    motifs_[name] = np.array(matrix).T
                    tf_names.append(name.split('.')[-1])
                name = line.split()[-1]
                matrix = []
            else:
                matrix.append([int(float(x)) for x in line.split()])
        if name:
            motifs_[name] = np.array(matrix).T
            tf_names.append(name.split('.')[-1])
    return motifs_, np.array(tf_names)


def pcc(pwm1, pwm2):
    pearson_score, _ = stats.pearsonr(np.ravel(pwm1), np.ravel(pwm2))
    return pearson_score


def align_and_calculate_score(pwm1, pwm2):
    """Slide the shorter PWM across the longer one (and its reverse complement)
    to find the best-aligned window, by Pearson correlation."""
    swap = False
    RC = False
    N = len(pwm1)
    M = len(pwm2)
    if M > N:
        print('Motif 2 > Motif 1', end=',')
        swap = True
        pwm1, pwm2 = pwm2, pwm1
        N, M = M, N

    dL = N - M + 1
    alignment_scores = np.zeros((dL, 2))
    for i in range(dL):
        alignment_scores[i, 0] = pcc(pwm1[i:i + M, :], pwm2)
        alignment_scores[i, 1] = pcc(pwm1[i:i + M, :], pwm2[::-1, ::-1])  # RC

    best_i = np.argmax(alignment_scores[:, 0])
    best_rc = np.argmax(alignment_scores[:, 1])
    if alignment_scores[best_rc, 1] > alignment_scores[best_i, 0]:
        best_i = best_rc
        RC = True
        pwm2 = pwm2[::-1, ::-1]
        print('RC', end=', ')

    aligned_pwm1 = pwm1[best_i:best_i + M, :]
    aligned_pwm2 = pwm2
    print(best_i)
    score = pcc(aligned_pwm1, aligned_pwm2)

    if swap and RC:
        pwm1, pwm2 = pwm2[::-1, ::-1], pwm1[::-1, ::-1]
    elif swap and not RC:
        pwm1, pwm2 = pwm2, pwm1

    return score, pwm1, pwm2, RC


def trimming(ppm):
    """Trim low-information-content flanks (IC <= 0.5) from a PPM."""
    IC = calculate_information_content(ppm)
    ikeep = np.where(IC > 0.5)[0]
    start, end = ikeep[0], ikeep[-1]
    return ppm[start:end + 1, :]


# ---------------------------------------------------------------------------
# 1. Load raw PFMs (HOCOMOCO + JASPAR) and the HOCOMOCO name -> gene mapping.
# ---------------------------------------------------------------------------

hoco_pfm_ = {}
current_motif = None
with open(f'{RAW_DIR}/H12CORE_pcms.txt', 'r') as f:
    for line in f:
        if line.startswith('>'):
            current_motif = line.strip()[1:]
            hoco_pfm_[current_motif] = []
        else:
            numbers = [int(float(x)) for x in line.strip().split('\t')]
            hoco_pfm_[current_motif].append(numbers)

with np.load(f'{INPUT_DIR}/HOCO_uniprot_to_gene.npz') as data:
    conversion = {key: data[key] for key in data.files}

hoco_motif_ = np.array(list(hoco_pfm_.keys()))
hoco_tfs_ = np.array([conversion[m.split('.')[0]].tolist() for m in hoco_motif_])

jasp_motif_, jasp_tfs_ = parse_motifs(f'{RAW_DIR}/JASPAR_2024_CORE_last_version_HomoSapiens.txt')

# ---------------------------------------------------------------------------
# 2. Keep only motifs for TFs expressed in LNCaP (DepMap Public 25Q2, TPM >= 3).
# ---------------------------------------------------------------------------

df_exp = pd.read_excel(f'{RAW_DIR}/Expression_Public_25Q2_LNCaP.xlsx')

pr_genes = df_exp.columns.values[1:]
pr_genes = np.array([gene.split(' ')[0] for gene in pr_genes])

exp = df_exp.iloc[0].values[1:]
exp_3_indx = np.where(exp >= 3)[0]
pr_genes_exp3 = pr_genes[exp_3_indx]

jasp_indx = []
for tf in pr_genes_exp3:
    if tf in jasp_tfs_:
        jasp_indx.append(np.where(jasp_tfs_ == tf)[0])  # if non-redundant, [0][0] and don't concatenate
jasp_indx = np.concatenate(jasp_indx)
jasp_mot = np.array(list(jasp_motif_.keys()))
jasp_subset_mot = jasp_mot[jasp_indx]
jasp_subset_pfm = {key: jasp_motif_[key] for key in jasp_subset_mot}

hoco_indx = []
for tf in pr_genes_exp3:
    if tf in hoco_tfs_:
        hoco_indx.append(np.where(hoco_tfs_ == tf)[0])
hoco_indx = np.concatenate(hoco_indx)
hoco_subset_mot = hoco_motif_[hoco_indx]
hoco_subset_pfm = {key: np.array(hoco_pfm_[key]) for key in hoco_subset_mot}

all_pfms = {**jasp_subset_pfm, **hoco_subset_pfm}

# ---------------------------------------------------------------------------
# 3. PFM -> PPM -> PWM, with pseudocount via a fixed A/C/G/T background.
# ---------------------------------------------------------------------------

bg_freq = np.array([0.30, 0.20, 0.20, 0.30]).reshape(1, 4)  # A, C, G, T

PPM = {}
PWM = {}
for motif in all_pfms.keys():
    a = np.array(all_pfms[motif]) + bg_freq
    ppm = a / np.vstack(np.sum(a, axis=1))
    pwm = np.log2(ppm / bg_freq)
    PPM[motif] = np.copy(ppm)
    PWM[motif] = np.copy(pwm)

# ---------------------------------------------------------------------------
# 4. Group motifs by TF (a TF can have multiple JASPAR/HOCOMOCO motifs).
# ---------------------------------------------------------------------------

motif_ids = list(PPM.keys())
TFs = []
for motif_id in motif_ids:
    if 'H12CORE' in motif_id:
        TFs.append(conversion[motif_id.split('.')[0]].tolist())
    else:
        TFs.append(motif_id.split('.')[-1])
TFs = np.array(TFs)

tf_motifs = {}
for tf in np.unique(TFs):
    motif_indices = np.where(TFs == tf)[0]
    tf_motifs[tf] = [motif_ids[i] for i in motif_indices]

# ---------------------------------------------------------------------------
# 5. For TFs with multiple motifs, auto-detect a consistent relative
#    orientation: align every motif in the group against the group's most
#    representative motif (highest total pairwise similarity), and flag which
#    ones need to be reverse-complemented (motifs_RC_dict).
# ---------------------------------------------------------------------------

mot_rc_names = []
mot_rc_results = []
for tf in np.unique(TFs):
    motifs_ = np.array(tf_motifs[tf])
    print(f'next tf {tf}, motifs: {len(motifs_)}')
    n_mot = len(motifs_)
    scores = np.zeros((n_mot, n_mot))
    for i in range(n_mot):
        for j in range(n_mot):
            pwm1 = trimming(PPM[motifs_[i]])
            pwm2 = trimming(PPM[motifs_[j]])
            score, _, _, _ = align_and_calculate_score(pwm1, pwm2)
            scores[i, j] = score

    imax = np.argmax(np.sum(scores, axis=1))
    motif_ref = motifs_[imax]
    pwm1 = trimming(PPM[motif_ref])

    for k in range(n_mot):
        motif_2 = motifs_[k]
        pwm2 = trimming(PPM[motif_2])
        _, _, _, RC_ = align_and_calculate_score(pwm1, pwm2)
        mot_rc_names.append(motif_2)
        mot_rc_results.append(RC_)

motifs_RC_dict = dict(zip(mot_rc_names, mot_rc_results))

for motif in PWM.keys():
    if motifs_RC_dict[motif]:
        PPM[motif] = PPM[motif][::-1, ::-1]
        PWM[motif] = PWM[motif][::-1, ::-1]

# ---------------------------------------------------------------------------
# 6. Visual QC: plot every TF-group's motifs, using the now-aligned PPM, so the
#    plots reflect the same orientation as the saved PWM. 
# ---------------------------------------------------------------------------

if RUN_VISUAL_QC:
    for tf in np.unique(TFs):
        print('------------------------------------')
        print(tf)
        for motif_2 in tf_motifs[tf]:
            logo(PPM[motif_2], motif_2)

# ---------------------------------------------------------------------------
# 7. Manual fix: the automatic alignment fails for CTCF specifically (its
#    motif is unusually long), so its two motifs are reverse-complemented by
#    hand, directly in PWM (the dict that gets saved).
# ---------------------------------------------------------------------------

PWM['MA0139.2.CTCF'] = PWM['MA0139.2.CTCF'][::-1, ::-1]
PWM['CTCF.H12CORE.0.P.B'] = PWM['CTCF.H12CORE.0.P.B'][::-1, ::-1]

np.savez(OUTPUT_PATH, **PWM)
print(f"Wrote {len(PWM)} motif PWMs to {OUTPUT_PATH}")
