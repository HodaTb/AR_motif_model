"""Build the mutagenesis analysis dataset (Methods 2.2.2) from raw barcode counts.

Required inputs:
  - ../raw_data/GSE335266_LNCaP_barcode_counts.tsv.gz -- raw DNA/RNA barcode counts
    per plasmid variant, from GEO accession GSE335266 (Tekoglu et al., manuscript in
    preparation; available to reviewers via a private GEO reviewer link, public upon
    publication). Not published in this repo. Columns: BC, DHT.R1, DHT.R2, DHT.R3,
    BCTOTAL, BCMAXMUT, VARIANTS -- DHT is the only measured RNA condition (no EtOH/
    vehicle-control replicates in this assay); BCMAXMUT is used as the per-variant
    library-size normalizer.
  - ../input_files/mutagenesis_region_reference.csv -- one row per mutagenized region,
    columns Region (name), Region_Tag (12-character tag identifying the region in the
    raw GEO barcode-count data's 'VARIANTS' field: 6 bp forward-primer prefix + the
    reverse complement of the 6 bp backward-primer prefix), and WT_seq (309 bp wild-
    type sequence). Included directly in this repo; the collaborator-provided primer
    table and region-coordinate file it was derived from are not.

Output: ../input_files/Complete_Mut_Data_LibVariant30_Pscount5.csv
(n_DNA >= 30 filter and pseudocount=5, per Methods 2.2.2; the '#_Plasmids > 2' and
variant-count filters applied on top of this are done later, in the model scripts
that consume this file.) LFC2WT_MLE is each variant's DHT signal relative to its
region's WT (unmutated) plasmid.
"""
import gzip
import numpy as np
import pandas as pd

RAW_DIR = '../raw_data'
REGION_REF_PATH = '../input_files/mutagenesis_region_reference.csv'
OUTPUT_PATH = '../input_files/Complete_Mut_Data_LibVariant30_Pscount5.csv'

PSEUDOCOUNT = 5

with gzip.open(f'{RAW_DIR}/GSE335266_LNCaP_barcode_counts.tsv.gz') as f:
    df_counts = pd.read_csv(f, sep='\t')

df_counts = df_counts[df_counts['BCMAXMUT'] > 29]  # n_DNA >= 30 filter

region_ref = pd.read_csv(REGION_REF_PATH)
regions = region_ref['Region'].values
region_tags = region_ref['Region_Tag'].values
region_seqs = region_ref['WT_seq'].values

var = df_counts['VARIANTS'].str.split(':').values
var_tags = np.array([var[i][0] for i in range(len(df_counts))])

final_df = pd.DataFrame()
for n in range(len(regions)):
    seq = region_seqs[n]

    df1 = df_counts[var_tags == region_tags[n]]
    df11 = df1.groupby(['VARIANTS']).sum()
    plasmid_num = df1.groupby(['VARIANTS']).size()

    dht = np.sum(df11[['DHT.R1', 'DHT.R2', 'DHT.R3']], axis=1) + PSEUDOCOUNT
    lib = df11['BCMAXMUT'].values + PSEUDOCOUNT
    dht_m = dht / lib
    dht_w = dht.iloc[-1] / lib[-1]  # last row is the WT (unmutated) plasmid

    lfc2wt = np.log2(dht_m / dht_w)

    df_rgn = pd.DataFrame({
        'Region': regions[n], 'Variants': df11.index, '#_Plasmids': plasmid_num,
        'DHT_Counts': dht_m, 'LFC2WT_MLE': lfc2wt, 'WT_seq': seq, 'Lib': lib,
    })

    # Reconstruct each variant's mutated sequence from its "pos:ref>alt ..." encoding.
    var_list = df_rgn['Variants'].str.split(' ').values
    all_mutated_seqs = []
    for i in range(len(var_list)):
        mutations = []
        for j in range(len(var_list[i]) - 1):
            new_base = var_list[i][j].split('>')[1]
            pos = int(var_list[i][j].split('>')[0].split(':')[1]) - 2
            if pos < 309:
                mutations.append((pos, new_base))

        seq_list = list(seq)
        for pos, new_base in mutations:
            seq_list[pos] = new_base
        all_mutated_seqs.append(''.join(seq_list))
    df_rgn['Mutated_seq'] = all_mutated_seqs

    final_df = pd.concat([final_df, df_rgn], ignore_index=True)

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(final_df)} variant records to {OUTPUT_PATH}")
