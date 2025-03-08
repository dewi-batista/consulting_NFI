import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load data
data = pd.read_csv('data/augmented_mixtures.csv')

# # histograms and density plots to check distribution shape
# for col in data.columns[1 : -1]:
#     plt.figure(figsize=(6,4))
#     sns.histplot(data[col].dropna(), bins=30, kde=True)
#     plt.title(f'{col}')
#     plt.xlabel('Value')
#     plt.ylabel('Freq.')
#     plt.show()

# correlations condition on:
data_blood = data[data['Blood'] == 1]

# blelele
marker_start_index = list(data.columns).index("HBB")
markers = data.columns[marker_start_index :]
corr_matrix = data_blood[markers].corr()

# plot that correlation matrix boiiii
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix for Marker Values (Blood = 1)")
plt.show()
