import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from correlation import get_clusters
from mpl_toolkits.mplot3d import Axes3D  # ?

# load data
data = pd.read_csv('data/preproc_mixtures.csv')

# all six fluid combinations present in mixtures.csv
fluid_combinations = [
    # ('Semen.fertile', 'Vaginal.mucosa'),
    # ('Saliva', 'Vaginal.mucosa'),
    # ('Blood', 'Nasal.mucosa'),
    # ('Nasal.mucosa', 'Saliva'),
    # ('Blood', 'Vaginal.mucosa'),
    ('Blood', 'Menstrual.secretion')
]

corr_threshold = 0.2
clusters = get_clusters(corr_threshold)

# set starting index of markers in csv
marker_start_index = list(data.columns).index("HBB")
markers = data.columns[marker_start_index:]
for (fluid_1, fluid_2) in fluid_combinations:

    # ...
    restricted_data = data[(data[fluid_1] == 1) & (data[fluid_2] == 1)]
    for correlates in clusters[f"{fluid_1}+{fluid_2}"]:
        print(correlates)

        marker_1 = correlates[0]
        marker_2 = correlates[1]

        # Compute a 2D histogram with 10 bins along each dimension
        hist, xedges, yedges = np.histogram2d(restricted_data[marker_1], restricted_data[marker_2], bins=10)

        # Create meshgrid for the bin positions
        xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1]-xedges[0])/2,
                                yedges[:-1] + (yedges[1]-yedges[0])/2)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # Define the width and depth of each bar
        dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
        dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
        dz = hist.flatten()

        # Plotting the 3D histogram
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        ax.set_xlabel(f"{marker_1}")
        ax.set_ylabel(f"{marker_2}")
        ax.set_zlabel('Freq.')
        ax.set_title(f"Distribution of {marker_1} against {marker_2} conditioned on {fluid_1}+{fluid_2} (correlation {np.round(correlates[-1], 2)})")
        plt.savefig(f"figures/3D_hists/conditioned_on_({fluid_1},{fluid_2})_distribution_of_({marker_1},{marker_2}).pdf")
