import numpy  as np
import pandas as pd

def augment(data, num_aug_samples, max_num_fluids):
    
    marker_start_index = list(data.columns).index("HBB")
    fluids = data.columns[: marker_start_index]

    augmented_data = []
    for _ in range(num_aug_samples):
        
        # choose fluids (randomly) to include in augmented sample
        fluids_to_include = np.random.choice(len(fluids), np.random.randint(1, max_num_fluids + 1), replace=False)

        # for each fluid to include, randomly sample a single row
        samples = []
        for fluid_index in fluids_to_include:
            samples.append(data[data.iloc[:, fluid_index] == 1].sample())
        mixture_of_samples = pd.concat(samples).reset_index(drop=True)

        # take each marker's max
        augmented_sample = mixture_of_samples.max().to_frame().T
        augmented_data.append(augmented_sample)

    # convert to pandas df
    augmented_data = pd.concat(augmented_data, ignore_index=True)

    # drop all columns right of PRM1
    markers_to_remove_start_index = list(augmented_data.columns).index("RPS4Y1")
    markers_to_remove = augmented_data.columns[markers_to_remove_start_index:]
    augmented_data = augmented_data.drop(columns=markers_to_remove)

    # drop Blank_PCR, Semen.sterile, Skin and Skin.penile
    augmented_data = augmented_data.drop(['Blank_PCR', 'Semen.sterile', 'Skin', 'Skin.penile'], axis=1)

    # save
    augmented_data.to_csv("data/augmented_mixtures.csv", index=False)

if __name__ == "__main__":

    # load data
    data = pd.read_csv("data/preproc_individuals.csv")

    # augment away!
    augmented_data = augment(data, num_aug_samples=10_000, max_num_fluids=2)
    
