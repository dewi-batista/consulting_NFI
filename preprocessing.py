import pandas as pd

# load data
csv_string = 'mixtures'
data = pd.read_csv(f'data/{csv_string}.csv')

# replace fluids column with its one-hot encoding
fluids_col = data.columns[0]

# how we preprocess differs slightly between individuals.csv and mixtures.csv
if csv_string == 'individuals':
    one_hotted_fluids = pd.get_dummies(data[fluids_col], dtype=int)
else:
    one_hotted_fluids = data[fluids_col].str.get_dummies(sep='+')

print(data[fluids_col].unique())

# # identify markers with low presence per fluid
# presence_threshold = 0.1
markers = data.columns[1:]
for marker in markers:
    rows_of_interest = data[data[fluids_col] == marker]
    print(rows_of_interest)

# replace the single fluids column with the one-hot encodings
data = data.drop(columns=[fluids_col])
data = pd.concat([one_hotted_fluids, data], axis=1)

# replace NaNs with 0s
data.fillna(0, inplace=True)

# drop replicate values column (subject to change)
data = data.iloc[:, :-1]

# save data ('%.0f' ensures that everything is int)
data.to_csv(f'data/preproc_{csv_string}.csv', index=False)#, float_format='%.0f')
