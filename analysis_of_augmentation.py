import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

# load data
data_mix = pd.read_csv('data/preproc_mixtures.csv')
data_aug = pd.read_csv('data/augmented_mixtures.csv')

# all six fluid combinations present in mixtures.csv
fluid_combinations = [
    ('Semen.fertile', 'Vaginal.mucosa'),
    ('Saliva', 'Vaginal.mucosa'),
    ('Blood', 'Nasal.mucosa'),
    ('Nasal.mucosa', 'Saliva'),
    ('Blood', 'Vaginal.mucosa'),
    ('Blood', 'Menstrual.secretion')
]

# set starting index of markers in csv
marker_start_index = list(data_mix.columns).index("HBB")
markers = data_mix.columns[marker_start_index:]
for (fluid_1, fluid_2) in fluid_combinations:
    print()
    print(fluid_1, fluid_2)
    for col in markers:
        restricted_data_mix = data_mix[(data_mix[fluid_1] == 1) & (data_mix[fluid_2] == 1)]
        restricted_data_aug = data_aug[(data_aug[fluid_1] == 1) & (data_aug[fluid_2] == 1)]
        if (restricted_data_mix[col] != 0).mean() <= 0.2:
            continue

        sample1 = restricted_data_mix[col]
        sample2 = restricted_data_aug[col]

        # the following is to account for uneqally sized samples
        min_size = min(len(sample1), len(sample2))

        # Randomly select a subset from the larger sample
        sample_1_sub = sample1.sample(n=min_size, random_state=42) if len(sample1) > len(sample2) else sample1
        sample_2_sub = sample2.sample(n=min_size, random_state=42) if len(sample2) > len(sample1) else sample2

        # Now, perform the KS test
        ks_stat, p_value = ks_2samp(sample_1_sub, sample_2_sub)
        print(f"{col} - P-Value: {np.round(p_value, 3)}")
