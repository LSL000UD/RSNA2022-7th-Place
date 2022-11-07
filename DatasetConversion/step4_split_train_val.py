import os
import pickle
import random

import path

if __name__ == '__main__':
    source_dir = f"{path.path_root}/CompetitionData/Image"

    list_train = []
    list_val = []

    for file in os.listdir(source_dir):
        if file.find('.nii.gz') == -1:
            continue

        case_id = file.split('.nii.gz')[0]
        if random.random() <= 0.25:
            list_val.append(case_id)
        else:
            list_train.append(case_id)

    pickle.dump({'train':list_train, 'val':list_val}, open('./data_split.pkl', 'wb'))