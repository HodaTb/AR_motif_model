"""Build the per-region Log2FC (DHT/EtOH) aggregation table (Methods, region-level
inducibility). Used only to annotate Table S3 / fig6_gwas_variants.ipynb.

Required raw inputs (place in ../raw_data/):
  - lncap-starrseq-counts-per-library-tile-stranded-v3.0.txt -- raw per-tile
    DHT/EtOH/lib counts and sequences (Huang et al. 2021)
  - tile.stranded-unique.bed -- per-tile genomic coordinates (BED6: Chromosome,
    Start, End, tile name matching `tilename` in the raw txt above, Score,
    Strand).

Also required (already in this repo:
  - ../input_files/regions_coords.csv -- region coordinates (Chromosome/Start/End/Name).
  - ../input_files/regions.fa -- region sequences, headers formatted
    `..._3545_<NAME> range=...`.

Output: ../input_files/regions_counts_table.csv

A region is assigned counts from tiles that fully cover its central 100bp (the
AR motif core, +-50bp around the region midpoint).
"""
import numpy as np
import pandas as pd
from Bio import SeqIO

RAW_DIR = '../raw_data'
INPUT_DIR = '../input_files'
OUTPUT_PATH = f'{INPUT_DIR}/regions_counts_table.csv'

LCUT = 5  # minimum lib value
PSEUDOCOUNT = 5
COUNT_COLS = ['dht1', 'dht2', 'dht3', 'etoh1', 'etoh2', 'etoh3']

# 1. Stream through the raw per-tile file once: accumulate genome-wide totals
#    (over every tile, for size-factor normalization below) and keep per-tile
#    rows that pass QC (lib >= LCUT, no 'N's).
totals = np.zeros(len(COUNT_COLS))
kept = {'Tile_name': [], 'lib': [], **{col: [] for col in COUNT_COLS}}

with open(f'{RAW_DIR}/lncap-starrseq-counts-per-library-tile-stranded-v3.0.txt') as fin:
    for line in fin:
        if line[0] != 't':  # skip the header row
            fields = line.split()
            counts = np.array(fields[2:8], dtype=float)
            totals += counts
            lib = float(fields[8])
            seq = fields[-1]
            if 'N' not in seq and lib >= LCUT:
                kept['Tile_name'].append(fields[0])
                kept['lib'].append(lib)
                for col, val in zip(COUNT_COLS, counts):
                    kept[col].append(val)

df_keep = pd.DataFrame(kept)

# Size-factor normalize the replicate counts (library-size correction, using
# totals over every tile, not just the QC-passing subset) before any
# aggregation, matching the original notebook.
sf = pd.Series(totals, index=COUNT_COLS)
sf = sf / sf.mean()
df_keep[COUNT_COLS] = df_keep[COUNT_COLS].div(sf, axis=1)

# Attach genomic coordinates.
tile_coords = pd.read_csv(f'{RAW_DIR}/tile.stranded-unique.bed', sep='\t', header=None,
                           names=['Chromosome', 'Start', 'End', 'Tile_name', 'Score', 'Strand'])
df_keep = df_keep.merge(tile_coords[['Tile_name', 'Chromosome', 'Start', 'End']], on='Tile_name', how='left')


# 2. For each region, find tiles that fully cover its central 100bp (AR motif core).
def find_covering_intervals(tile_i, tile_f, a, b):
    covering_indices = np.zeros(len(tile_i))
    for index, (start, end) in enumerate(zip(tile_i, tile_f)):
        if start <= a and end >= b:
            covering_indices[index] = 1
    return covering_indices


regions = pd.read_csv(f'{INPUT_DIR}/regions_coords.csv')
region_name = regions['Name'].values
region_chr = regions['Chromosome'].values
region_start = regions['Start'].values
region_end = regions['End'].values

rgn_dht = np.zeros(len(regions))
rgn_etoh = np.zeros(len(regions))
rgn_lf2 = np.zeros(len(regions))
n_tiles_covering = []

for i, rgn in enumerate(region_name):
    df_chr = df_keep[df_keep['Chromosome'] == region_chr[i]]
    tile_i = df_chr['Start'].values
    tile_f = df_chr['End'].values

    rgn_i, rgn_f = region_start[i], region_end[i]
    ar_i, ar_f = int((rgn_i + rgn_f) / 2 - 50), int((rgn_i + rgn_f) / 2 + 50)

    cover_indx = find_covering_intervals(tile_i, tile_f, ar_i, ar_f)
    df_overlap = df_chr[cover_indx.astype(bool)]
    n = len(df_overlap)
    libs = df_overlap['lib'].sum()
    dht = df_overlap[['dht1', 'dht2', 'dht3']].mean(axis=1).sum() + PSEUDOCOUNT
    etoh = df_overlap[['etoh1', 'etoh2', 'etoh3']].mean(axis=1).sum() + PSEUDOCOUNT

    rgn_dht[i] = (dht / n) / libs
    rgn_etoh[i] = (etoh / n) / libs
    rgn_lf2[i] = np.log2(dht / etoh)
    n_tiles_covering.append(n)

    print(i, rgn, n)

regions['region_DHT'] = rgn_dht
regions['region_EtOH'] = rgn_etoh
regions['region_Log2FC'] = rgn_lf2
regions['n_tiles_covering'] = n_tiles_covering


# 3. Attach region sequences from regions.fa.
def parse_fasta(file_path):
    names = []
    sequences = []
    for record in SeqIO.parse(file_path, 'fasta'):
        names.append(record.id.split('3545_')[1])
        sequences.append(str(record.seq))
    return pd.DataFrame({'Name': names, 'Sequence': sequences})


sequences_df = parse_fasta(f'{INPUT_DIR}/regions.fa')
grouped = regions.merge(sequences_df, on='Name', how='left')

grouped.to_csv(OUTPUT_PATH)
print(f"Wrote {len(grouped)} regions to {OUTPUT_PATH}")
