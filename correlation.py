import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data = pd.read_csv('data/preproc_mixtures.csv')

fluid_combinations = [
    ('Semen.fertile', 'Vaginal.mucosa'),
    # ('Saliva', 'Vaginal.mucosa'),
    # ('Blood', 'Nasal.mucosa'),
    # ('Nasal.mucosa', 'Saliva'),
    # ('Blood', 'Vaginal.mucosa'),
    # ('Blood', 'Menstrual.secretion')
]

# set starting index of markers in csv
marker_start_index = list(data.columns).index("HBB")
markers = data.columns[marker_start_index :]

# compute correlations for all possible fluid couples
corr_threshold = 0.2
max_num_correlates = 2
for (fluid_1, fluid_2) in fluid_combinations:
    
    # restrict dataset to just the rows where these two fluids appear
    restricted_data = data[(data[fluid_1] == 1) & (data[fluid_2] == 1)]

    # compute correlation matrix
    corr = restricted_data[markers].loc[:, (restricted_data[markers] != 0).any(axis=0)].corr()

    # only consider markers that actually exist in the correlation matrix
    non_zero_markers = corr.index.tolist()
    print(fluid_1, fluid_2)
    for marker in non_zero_markers:
        top_n_correlates = corr.loc[marker].abs().nlargest(max_num_correlates + 1)
        top_n_correlates = top_n_correlates[top_n_correlates > corr_threshold].index.tolist()
        if marker in top_n_correlates:
            top_n_correlates.remove(marker)
        print(marker, top_n_correlates)
    print()

    # plot that correlation matrix boiiii
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation matrix of markers for {fluid_1} and {fluid_2}")
    plt.show()
