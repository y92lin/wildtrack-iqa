import torch
import pandas
import numpy as np
from torch.utils.data import Dataset


class WildTrackDataset(Dataset):
    categories = {'Bongo': 0, 'Cheetah': 1, 'Elephant': 2, 'Jaguar': 3, 'Leopard': 4, 'Lion': 5, 'Otter': 6, 'Panda': 7,
                  'Puma': 8, 'Rhino': 9, 'Tapir': 10, 'Tiger': 11}

    def __init__(self, dataset_file, binary_threshold: 4):
        images = pandas.read_csv(dataset_file)
        brisque_columns = ['feature_' + str(i) for i in range(0, 36)]
        feature_columns = brisque_columns + ['n_score', 'p_score', 'species']
        image_features = images[feature_columns]
        image_features['species'] = images['species'].apply(lambda x: self.categories[x])
        self.row_count = images.shape[0]

        # get rating
        self.mos = images["subj_score"].to_numpy()
        self.mos = np.select([(self.mos < binary_threshold), (self.mos >= binary_threshold)],[0,1])
        self.features = []
        self.label = []

        for index, row in image_features.iterrows():
            self.features.append(row.to_numpy())
            self.label.append(self.mos[index])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (torch.Tensor([self.label[idx]]), torch.Tensor(self.features[idx]))