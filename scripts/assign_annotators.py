import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Assign pairs to annotators and build annotation sheets.')
parser.add_argument('--pairs',  default='results_data/pairs_locked.csv',
                    help='Path to pairs_locked.csv')
parser.add_argument('--claims', default='Phase 3/claims.csv',
                    help='Path to claims.csv (output of extract_claims.py)')
parser.add_argument('--outdir', default='Phase 3/',
                    help='Directory to write annotation CSVs and pair_assignments.csv')
args = parser.parse_args()

pairs  = pd.read_csv(args.pairs)
claims = pd.read_csv(args.claims)

rng = np.random.default_rng(seed=1337)  # distinct from pairs_locked's seed=42, documented separately

assignments = []
for eth in pairs['ethnicity'].unique():
    for gt in ['genuine', 'impostor']:
        sub = pairs[(pairs.ethnicity==eth) & (pairs.ground_truth_label==gt)]['pair_id'].tolist()
        rng.shuffle(sub)
        # 5 kappa, 12 annotator_A, 13 annotator_B out of 30 (5+12+13=30)
        kappa = sub[:5]
        a = sub[5:17]
        b = sub[17:30]
        for pid in kappa:
            assignments.append({'pair_id': pid, 'group': 'kappa_subset'})
        for pid in a:
            assignments.append({'pair_id': pid, 'group': 'annotator_A'})
        for pid in b:
            assignments.append({'pair_id': pid, 'group': 'annotator_B'})

os.makedirs(args.outdir, exist_ok=True)
assign_df = pd.DataFrame(assignments)
assign_df.to_csv(os.path.join(args.outdir, 'pair_assignments.csv'), index=False)

print(assign_df['group'].value_counts())
merged = pairs.merge(assign_df, on='pair_id')
print(merged.groupby(['group', 'ethnicity']).size())

# Now build the actual claim-level annotation sheets
claims_a     = claims.merge(assign_df[assign_df.group == 'annotator_A'][['pair_id']], on='pair_id')
claims_b     = claims.merge(assign_df[assign_df.group == 'annotator_B'][['pair_id']], on='pair_id')
claims_kappa = claims.merge(assign_df[assign_df.group == 'kappa_subset'][['pair_id']], on='pair_id')

for name, df in [('annotator_A', claims_a), ('annotator_B', claims_b), ('kappa_shared', claims_kappa)]:
    df = df.sort_values(['pair_id', 'model_name', 'prompt_variant']).reset_index(drop=True)
    if name == 'kappa_shared':
        df['label_chics'] = ''
        df['label_friend'] = ''
    else:
        df['label'] = ''
    df['notes'] = ''
    out_path = os.path.join(args.outdir, f'annotation_{name}.csv')
    df.to_csv(out_path, index=False)
    print(f"{name}: {len(df)} claims across {df.pair_id.nunique()} pairs → {out_path}")
