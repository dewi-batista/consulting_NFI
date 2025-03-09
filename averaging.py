import numpy as np
import pandas as pd

# load data
data = pd.read_csv('data/preproc_mixtures.csv')

# print(list(data.iloc[0, 0:6]))

# ?
temp = data[(data['Semen.fertile'] == 1) & (data['Vaginal.mucosa'] == 1)]
marker_start_index = list(data.columns).index("HBB")
markers = data.columns[marker_start_index: ]
mean_list = temp[markers].mean().to_list()
print([round(x, 0) for x in mean_list] )

aug_data = pd.read_csv('data/augmented_mixtures.csv')

temp = aug_data[(aug_data['Semen.fertile'] == 1) & (aug_data['Vaginal.mucosa'] == 1)]
markers = aug_data.columns[marker_start_index: ]
print(temp[markers])