import pandas as pd

from pgmpy.estimators import ExhaustiveSearch

# load data
data = pd.read_csv('data/preproc_mixtures.csv')

# all six fluid combinations present in mixtures.csv
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

(fluid_1, fluid_2) = fluid_combinations[0]
restricted_data = data[(data[fluid_1] == 1) & (data[fluid_2] == 1)]
restricted_data['MUC4', 'MYOZ1', 'CYP2B7P1']

es = ExhaustiveSearch(restricted_data, scoring_method='bic')
best_model = es.estimate()
print(best_model.edges())

print("\nAll DAGs by score:")
for score, dag in reversed(es.all_scores()):
    print(score, dag.edges())