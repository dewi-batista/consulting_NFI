import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data = pd.read_csv('data/preproc_mixtures.csv')

# # histograms and density plots to check distribution shape
# for col in data.columns[1 : -1]:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data[col].dropna(), bins=30, kde=True)
#     plt.title(f'{col}')
#     plt.xlabel('Value')
#     plt.ylabel('Freq.')
#     plt.show()

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
max_num_correlates = 2
corr_threshold = 0.2
for (fluid_1, fluid_2) in fluid_combinations:
    
    # restrict dataset to just the rows where these two fluids appear
    restricted_data = data[(data[fluid_1] == 1) & (data[fluid_2] == 1)]

    # compute correlation matrix
    corr = restricted_data[markers].corr()

    print()
    print(fluid_1, fluid_2)

    # get markers that best correlate (non-negligibly) with each other marker
    for marker in markers:
        top_n_correlates = corr.loc[marker].abs().nlargest(max_num_correlates + 1)
        top_n_correlates = top_n_correlates[top_n_correlates > corr_threshold].index.tolist()
        if marker in top_n_correlates: # get rid of marker (correlates with itself)
            top_n_correlates.remove(marker)
        print(marker, top_n_correlates)

    # plot that correlation matrix boiiii
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
