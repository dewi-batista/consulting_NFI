import pandas as pd

# load data
csv_string = 'individuals'
data = pd.read_csv(f'data/{csv_string}.csv')

# replace fluids column with its one-hot encoding
fluids_col = data.columns[0]

# this needs to be modified for mixtures.csv
one_hotted_fluids = pd.get_dummies(data[fluids_col], dtype=int)
data = data.drop(columns=[fluids_col])
data = pd.concat([one_hotted_fluids, data], axis=1)

# replace NaNs with 0s
data.fillna(0, inplace=True)

# drop replicate values column (subject to change)
data = data.iloc[:, :-1]

# save data ('%.0f' ensures that everything is int)
data.to_csv(f'data/preprop_{csv_string}.csv', index=False)#, float_format='%.0f')
