import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from pdb import set_trace

#' Try to obtain the peak heights distribution as a mixture of Gaussians. 
#' Try to obtain it for simple cases first: one sample type, one marker
#' Then try to generalize to multiple mixture types

def gauss_mixture_fit(data_set, idx_marker, idx_sample, 
                       n_components=1, covariance_type="diag",
                       plot_=True, do_model_selection=True):
    """
    This function fits a mixture of Gaussians. If enabled, it will plot the histogram overlapped
    with the fitted distribution, as well as perform model selection for the optimal number of components.

    --- Inputs:

    data_set = string. If it equals "individuals", then it uses the data set from individual fluids.
    Otherwise it uses the mixtures

    idx_marker = integer. The index of the relevant marker in the list of markers from the data
    The list:

    - TBD: 

    Implement model selection per number of components
    Implement data generation
    Test via the KS test the adequacy of the fit
    """
    
    # - Load data
    if data_set == "individuals":
        data = pd.read_csv('data/individuals.csv').fillna(0)
    else:
        data = pd.read_csv('data/mixtures.csv').fillna(0)

    markers = data.columns[1:-5]
    samples = data.iloc[:, 0].unique()

    data_sample = data[data.iloc[:, 0] == samples[idx_sample]]

    # - Fit a mixture of gaussians
    
    #' Include model selection per number of components
    #' BIC favours simpler models, AIC


    model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                            n_init=3, random_state=0).fit(data_sample.loc[:, [markers[idx_marker]]])

    print(f"Convergence status: {model.converged_}")

    print(f"Weights: {model.weights_}")
    print(f"Means: {model.means_}")
    print(f"Covariances: {model.covariances_}")

    # - Plot the fitted curve:
    if plot_ == True:
        # Generate the curve

        x_val = np.arange(data_sample[markers[idx_marker]].min(), data_sample[markers[idx_marker]].max(), 1) 
        y_val = np.zeros((len(model.weights_), x_val.shape[0]))
        y_all = np.zeros(x_val.shape)

        # Compute each individual contribution:
        for i in range(len(model.weights_)):
            weight_ = model.weights_[i]
            mean_ = model.means_[i]

            if covariance_type == "full":
                covar_ = model.covariances_[i] 
            elif covariance_type == "spherical":
                covar_ = model.covariances_[i]
            elif covariance_type == "tied":
                covar_ = model.covariances_
            else:
                covar_ = np.diag(model.covariances_[i])
                

            y_val[i, :] = weight_ * multivariate_normal.pdf(x_val, mean=mean_, cov=covar_)
            y_all = y_all + y_val[i, :]


        plt.figure(figsize=(6,4))

        plt.hist(data_sample[markers[idx_marker]], bins=30, label="True Data", density=True)
        plt.plot(x_val, y_all, color="blue", label="Sum Contrib")

        for i in range(len(model.weights_)):
            plt.plot(x_val, y_val[i, :], label=f"Component {i}")

        plt.title(f'{samples[idx_sample]} - {markers[idx_marker]}\nTrue data and fit')
        plt.xlabel('Value')
        plt.ylabel('Freq.')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # --- Print the available choices

    #' Mixed samples: ['Semen.fertile+Vaginal.mucosa' 'Saliva+Semen.fertile'
    #'  'Saliva+Vaginal.mucosa' 'Blood+Nasal.mucosa' 'Nasal.mucosa+Saliva'
    #'  'Vaginal.mucosa+Blood' 'Menstrual.secretion+Blood']

    #' Individual samples: ['Blood' 'Saliva' 'Vaginal.mucosa' 'Menstrual.secretion' 'Semen.fertile'
    #' 'Semen.sterile' 'Nasal.mucosa' 'Blank_PCR' 'Skin' 'Skin.penile']

    #' Markers: [HBB,ALAS2,CD93,HTN3,STATH,BPIFA1,MUC4,MYOZ1,CYP2B7P1,MMP10,MMP7,MMP11,SEMG1,KLK3,PRM1]

    # --- Check the first markers for blood and nasal mucosa

    #' Start with a marker which separates the samples very well: HBB

    idx_marker = 0    # Check HBB

    gauss_mixture_fit("individuals", idx_marker, 0, n_components=2, covariance_type="diag")
    gauss_mixture_fit("individuals", idx_marker, 6, n_components=1, covariance_type="diag", plot_=False)

    #set_trace()
    gauss_mixture_fit("mixtures", idx_marker, 3, n_components=3, covariance_type="diag")