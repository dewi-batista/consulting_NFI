import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from correlation import get_clusters
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

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
        
        # just the first one for now while I test
        if correlates[0] != 'MUC4':
            continue        
        print(correlates)

        # assign corelative markers
        marker_1 = correlates[0]
        marker_2 = correlates[1]

        # assign relevant marker data to fit Gaussian mixture and plot hist.
        markers_data = restricted_data[[marker_1, marker_2]]

        # fit 2D Gaussian mixture with two components
        gmm = GaussianMixture(n_components=2, covariance_type="full")
        # gmm.fit(markers_data)

        scaled_data = scaler.fit_transform(markers_data)
        gmm.fit(scaled_data)

        # =========================================================

        # 2D hist with 10 bins
        hist, xedges, yedges = np.histogram2d(restricted_data[marker_1], restricted_data[marker_2], bins=10)
        # hist, xedges, yedges = np.histogram2d(scaled_data[marker_1], scaled_data[marker_2], bins=10)

        # meshgrid for the bin positions
        xpos, ypos = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2, yedges[:-1] + (yedges[1] - yedges[0]) / 2)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # width and depth of each bar
        dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
        dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
        dz = hist.flatten()

        # plot Gaussian mixture time
        x_vals = np.linspace(xedges[0], xedges[-1], 50)
        y_vals = np.linspace(yedges[0], yedges[-1], 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        xy_grid = np.column_stack([X.ravel(), Y.ravel()])

        # compute GMM density
        pdf_vals = np.exp(gmm.score_samples(xy_grid))
        Z_pdf = pdf_vals.reshape(X.shape)

        n_points = markers_data.shape[0]
        bin_area = (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
        Z_scaled = Z_pdf * n_points * bin_area

        # plot time babyyyyy
        fig = plt.figure(figsize=(10, 7))
        
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

        # this is for gm
        ax.plot_surface(X, Y, Z_scaled, alpha=0.5, color='red')

        ax.set_xlabel(f"{marker_1}")
        ax.set_ylabel(f"{marker_2}")
        ax.set_zlabel('Freq.')
        ax.set_title(f"Distribution of {marker_1} against {marker_2} conditioned on {fluid_1}+{fluid_2} (correlation {np.round(correlates[-1], 2)})")
        # plt.savefig(f"figures/3D_hists/conditioned_on_({fluid_1},{fluid_2})_distribution_of_({marker_1},{marker_2}).pdf")
        plt.show()
