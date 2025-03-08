import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data = pd.read_csv('data/mixtures.csv')

# histograms and density plots to check distribution shape
for col in data.columns[1 : -1]:
    plt.figure(figsize=(6,4))
    sns.histplot(data[col].dropna(), bins=30, kde=True)
    plt.title(f'{col}')
    plt.xlabel('Value')
    plt.ylabel('Freq.')
    plt.show()
    