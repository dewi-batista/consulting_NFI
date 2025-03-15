# Good for all
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# For stats things
from scipy.stats import multivariate_normal, ks_2samp
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Python quality of life
from pdb import set_trace
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def gauss_mixture_fit(data_set, idx_marker, idx_sample, 
                       n_components=1, covariance_type="diag",
                       plot_=True, do_model_selection=False):
    """
    This function fits a mixture of Gaussians. If enabled, it will plot the histogram overlapped
    with the fitted distribution, as well as perform model selection for the optimal number of components.

    The documentation for GaussianMixture: 
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

    --- Inputs:

    data_set = string. If it equals "individuals", then it uses the data set from individual fluids.
    Otherwise it uses the mixtures

    idx_marker = integer. The index of the relevant marker in the list of markers from the data
    The list: [HBB,ALAS2,CD93,HTN3,STATH,BPIFA1,MUC4,MYOZ1,CYP2B7P1,MMP10,MMP7,MMP11,SEMG1,KLK3,PRM1]

    idx_sample = integer. The index of the relevant fluid type in the list of fluids. The list:
    - individuals: ['Blood' 'Saliva' 'Vaginal.mucosa' 'Menstrual.secretion' 'Semen.fertile'
    'Semen.sterile' 'Nasal.mucosa' 'Blank_PCR' 'Skin' 'Skin.penile']
    - mixtures: ['Semen.fertile+Vaginal.mucosa' 'Saliva+Semen.fertile'
    'Saliva+Vaginal.mucosa' 'Blood+Nasal.mucosa' 'Nasal.mucosa+Saliva'
    'Vaginal.mucosa+Blood' 'Menstrual.secretion+Blood']

    n_components = integer. The number of components to be used for the Gaussian mixture, set by the user.
    Relevant only if 'do_model_selection' is set to False

    covariance_type = string. Choose between ["diag", "full", "spherical", "tied"]. It gives the 
    type of matrix to be used in the mixture definitions. 

    plot_ = boolean. If true, it also plots a histogram of the relevant data together with the best mixture fit

    do_model_selection = boolean. If true, it performs model selection to find the best number of mixture components.
    **Note**: it uses BIC, which is the stricter criterion. It also has a maximum mixture number of 10. If those need to 
    be updated, this is an avenue of improvement


    --- Output:

    model = model class of Gaussian Mixtures. It returns the model with the best number of components (user specified or via BIC)

    --- TBD: 

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
    X = data_sample.loc[:, [markers[idx_marker]]]

    if do_model_selection == False:
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, 
                                n_init=3, random_state=0).fit(X) 
        # print(f"Number of components: {n_components}")
    else:
        max_components = 10
        BIC_list = np.zeros(max_components)

        for i in range(1, max_components + 1):
            model_temp = GaussianMixture(n_components=i, covariance_type=covariance_type, 
                                n_init=3, random_state=0).fit(X) 
            BIC_list[i-1] = model_temp.bic(X)

        # Select the model with the smallest BIC

        best_no_comp = np.argmin(BIC_list) + 1
        model = GaussianMixture(n_components=best_no_comp, covariance_type=covariance_type, 
                                n_init=3, random_state=0).fit(X)
        # print(f"Number of components: {best_no_comp}") 

    # print(f"Convergence status: {model.converged_}")

    # print(f"Weights: {model.weights_}")
    # print(f"Means: {model.means_}")
    # print(f"Covariances: {model.covariances_}")

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
            plt.plot(x_val, y_val[i, :], label=f"Component {i+1}")

        plt.title(f'{samples[idx_sample]} - {markers[idx_marker]}\nTrue data and fit')
        plt.xlabel('Value')
        plt.ylabel('Freq.')
        plt.legend()
        plt.show()
    
    return model

def data_generator(n, model, covariance_type="diag", threshold=150, seed=42):
    """
    This function generates data artificially, following the Gaussian mixture model from 'model'.
    The number of samples generated is 'n'
    The function also implements a thresolding mechanic: if a generated sample would be below 150, it is 
    returned as zero
    """

    np.random.seed(seed)

    # Initialize
    weights = model.weights_
    new_samples = np.zeros(n)
    temp1 = 0

    # Sample mixture components to take from 
    temp2 = np.random.choice(range(len(weights)), size=n, p=weights)
    mixture_number = np.bincount(temp2) 

    # Sample values from mixtures
    for i in range(len(weights)):
        # Extract curve parameters for relevant mixture component
        mean_ = model.means_[i]

        if covariance_type == "diag":
            covar_ = np.diag(model.covariances_[i])
        elif covariance_type == "full":
            covar_ = model.covariances_[i] 
        elif covariance_type == "spherical":
            covar_ = model.covariances_[i]
        else:
            covar_ = model.covariances_
        
        # Sample
        new_samples[temp1:(temp1+mixture_number[i])] = multivariate_normal.rvs(mean=mean_, cov=covar_, size=mixture_number[i])
        temp1 = temp1 + mixture_number[i]

    # Do thresholding
    new_samples[new_samples <= threshold] = 0

    return new_samples

if __name__ == "__main__":
    # --- Test mixture fitting:

    # - Blood (0), HBB (0)
    #gauss_mixture_fit("individuals", 0, 0, n_components=3, covariance_type="diag") 
    #gauss_mixture_fit("individuals", 0, 0, do_model_selection=True, covariance_type="diag")

    # - Saliva (1), STATH (4)
    #gauss_mixture_fit("individuals", 4, 1, do_model_selection=True, covariance_type="diag")
    
    # - Semen.fertile+Vaginal.mucosa (0), MUC4 (6)
    #gauss_mixture_fit("mixtures", 0, 6, do_model_selection=True, covariance_type="diag")

    # --- Test data generation
    
    idx_marker = 7
    idx_sample = 0
    
    # - Get data

    # Extract true data
    data = pd.read_csv('data/mixtures.csv').fillna(0)

    markers = data.columns[1:-5]
    fluid_combinations = data.iloc[:, 0].unique()

    data_sample = data[data.iloc[:, 0] == fluid_combinations[idx_sample]]
    data_sample, test_data_sample = train_test_split(data_sample, test_size=0.2, random_state=42)
    # print(data_sample)

    true_data = data_sample.loc[:, [markers[idx_marker]]]
    test_true_data = test_data_sample.loc[:, [markers[idx_marker]]]
    # print(true_data.shape, test_true_data.shape)
    print(true_data)

    # Create new data
    model = gauss_mixture_fit("individuals", 0, 0, do_model_selection=True, covariance_type="diag", plot_=False)
    new_data = data_generator(10**2, model, threshold=150)
    # print(new_data.shape)
    # print(true_data)
    # - Test equality of distributions

    result = ks_2samp(test_true_data.to_numpy().flatten(), new_data)
    print(f"p-value for two-sided KS test: {result.pvalue}")

    # - Plot the result 

    # Create the mixture curves
    x_val = np.arange(data_sample[markers[idx_marker]].min(), data_sample[markers[idx_marker]].max(), 1) 
    y_val = np.zeros((len(model.weights_), x_val.shape[0]))
    y_all = np.zeros(x_val.shape)

    # Compute each individual contribution:
    for i in range(len(model.weights_)):
        weight_ = model.weights_[i]
        mean_ = model.means_[i]
        covar_ = np.diag(model.covariances_[i])

        y_val[i, :] = weight_ * multivariate_normal.pdf(x_val, mean=mean_, cov=covar_)
        y_all = y_all + y_val[i, :]

    plt.figure(figsize=(6,4))

    plt.hist(new_data, bins=100, label="Generated Data", density=True, color="red", alpha=0.5)
    plt.hist(test_true_data, bins=50, label="Test Data", density=True, color="blue", alpha=0.5)
    plt.hist(true_data, bins=50, label="Train Data", density=True, color="green", alpha=0.5)
    
    plt.plot(x_val, y_all, color="black", label="Sum Contrib")

    for i in range(len(model.weights_)):
        plt.plot(x_val, y_val[i, :], label=f"Component {i+1}")

    plt.title(f'{fluid_combinations[idx_sample]} - {markers[idx_marker]}\nTrue data and generated data')
    plt.xlabel('Value')
    plt.ylabel('Freq.')
    plt.legend()
    plt.show()
    