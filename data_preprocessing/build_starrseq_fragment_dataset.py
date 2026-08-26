"""Build the STARR-seq tile-fit training dataset (Methods 2.2.1) by filtering,
balancing, and region-excluding the raw per-tile counts.

Required raw inputs (place in ../raw_data/):
  - lncap-starrseq-counts-per-library-tile-stranded-v3.0.txt -- raw per-tile
    DHT/EtOH/lib counts and sequences (Huang et al. 2021).
  - tile.stranded-unique.bed -- per-tile genomic coordinates (BED6: Chromosome,
    Start, End, tile name matching `tilename` in the raw txt above, Score,
    Strand).

Also required (already in this repo):
  - ../input_files/regions_coords.csv -- region coordinates (Chromosome/Start/End/Name).
  - ../input_files/mutagenesis_region_reference.csv -- used only for its Region
    column, to exclude the 40 regions tested in the saturation-mutagenesis
    experiment from tile-fit training.

Output: ../input_files/Final_df_40excl_balanced.csv

Tile_dht/Tile_etoh are stored as the tile's raw per-replicate mean DHT/EtOH
counts (no pseudocount, no library normalization) -- matching how
motif_model/data_preps.py's TileDataset consumes them
(`3*Tile_dht + pseudo_count`, reconstructing the 3-replicate sum with
pseudocount at load time).
"""
import numpy as np
import pandas as pd

RAW_DIR = '../raw_data'
INPUT_DIR = '../input_files'
OUTPUT_PATH = f'{INPUT_DIR}/Final_df_40excl_balanced.csv'

LCUT = 30       # minimum lib value
LMIN = 400      # min tile length
LMAX = 700      # max tile length
LOGFC_THR = 0
PSEUDOCOUNT = 5.  # for the QC logFC filter below only (not stored in the output)

# ---------------------------------------------------------------------------
# 1. Stream through the raw per-tile file and keep tiles that pass QC: no 'N'
#    in the sequence, lib >= LCUT, LMIN <= length <= LMAX, and logFC >= LOGFC_THR.
# ---------------------------------------------------------------------------

kept = {'Tile_name': [], 'Sequence': [], 'Tile_dht': [], 'Tile_etoh': [], 'lib': []}
with open(f'{RAW_DIR}/lncap-starrseq-counts-per-library-tile-stranded-v3.0.txt') as fin:
    for line in fin:
        if line[0] != 't':  # skip the header row
            fields = line.split()
            lib = float(fields[8])
            seq = fields[-1]
            lseq = len(seq)

            if 'N' not in seq and lib >= LCUT and LMIN <= lseq <= LMAX:
                dht_reps = np.array(fields[2:5], dtype=float)
                etoh_reps = np.array(fields[5:8], dtype=float)
                logfc = np.log((dht_reps.sum() + PSEUDOCOUNT) / (etoh_reps.sum() + PSEUDOCOUNT))

                if logfc >= LOGFC_THR:
                    kept['Tile_name'].append(fields[0])
                    kept['Sequence'].append(seq)
                    kept['Tile_dht'].append(dht_reps.mean())
                    kept['Tile_etoh'].append(etoh_reps.mean())
                    kept['lib'].append(lib)

df_kept = pd.DataFrame(kept)
print(f'{len(df_kept)} filtered tiles')

# ---------------------------------------------------------------------------
# 2. Attach genomic coordinates and assign each tile to the region it overlaps
#    by the largest number of bp; tiles overlapping no region are dropped.
# ---------------------------------------------------------------------------

tile_coords = pd.read_csv(f'{RAW_DIR}/tile.stranded-unique.bed', sep='\t', header=None,
                           names=['Chromosome', 'Start', 'End', 'Tile_name', 'Score', 'Strand'])
df_kept = df_kept.merge(tile_coords[['Tile_name', 'Chromosome', 'Start', 'End']], on='Tile_name', how='left')

regions = pd.read_csv(f'{INPUT_DIR}/regions_coords.csv')

# Indexed by df_kept's own row index (not by loop/group order) so results land
# back on the correct rows regardless of groupby's chromosome iteration order.
region_name_by_idx = pd.Series(index=df_kept.index, dtype=object)
overlap_bp_by_idx = pd.Series(index=df_kept.index, dtype=float)

for chrom, group in df_kept.groupby('Chromosome'):
    rgn_chr = regions[regions['Chromosome'] == chrom]
    if len(rgn_chr) == 0:
        continue

    t_start = group['Start'].values[:, None]
    t_end = group['End'].values[:, None]
    r_start = rgn_chr['Start'].values[None, :]
    r_end = rgn_chr['End'].values[None, :]

    overlap_bp = np.clip(np.minimum(t_end, r_end) - np.maximum(t_start, r_start), 0, None)

    best = np.argmax(overlap_bp, axis=1)
    best_overlap = overlap_bp[np.arange(len(group)), best]
    rgn_names = rgn_chr['Name'].values

    mask = best_overlap > 0
    region_name_by_idx.loc[group.index[mask]] = rgn_names[best[mask]]
    overlap_bp_by_idx.loc[group.index[mask]] = best_overlap[mask]

df_kept['Region_name'] = region_name_by_idx
df_kept['Overlap_bp'] = overlap_bp_by_idx
df_kept = df_kept[df_kept['Region_name'].notna()].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Balance tiles per region: resample every region to the (dataset-wide)
#    average number of tiles per region, upsampling regions with fewer tiles
#    and downsampling regions with more.
# ---------------------------------------------------------------------------

average_tiles = len(df_kept) / df_kept['Region_name'].nunique()

upsampled_df = pd.DataFrame()
downsampled_df = pd.DataFrame()
for region, group in df_kept.groupby('Region_name'):
    num_tiles = len(group)
    if num_tiles < average_tiles:
        upsampled_group = group.sample(n=int(average_tiles), replace=True, random_state=42)
        upsampled_df = pd.concat([upsampled_df, upsampled_group])
    elif num_tiles > average_tiles:
        downsampled_group = group.sample(n=int(average_tiles), random_state=42)
        downsampled_df = pd.concat([downsampled_df, downsampled_group])

final_df = pd.concat([upsampled_df, downsampled_df]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Exclude the 40 regions used in the saturation-mutagenesis experiment, so
#    it remains an independent held-out validation set.
# ---------------------------------------------------------------------------

mut_regions = pd.read_csv(f'{INPUT_DIR}/mutagenesis_region_reference.csv')['Region'].values
mut_regions = mut_regions + np.array(['-ARBS'] * len(mut_regions))
final_df = final_df[~final_df['Region_name'].isin(mut_regions)]

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(final_df)} tiles to {OUTPUT_PATH}")
