import re, os
from glob import glob
import config as con
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

la_enc = LabelEncoder()
c_int = la_enc.fit_transform(sorted(os.listdir(con.dataset_path)))
oh_enc = OneHotEncoder(sparse=False)
oh_enc.fit(np.array(c_int).reshape(-1, 1))


def encode_label(label):
    int_encoded = la_enc.transform(label)
    return oh_enc.transform(np.array(int_encoded).reshape(-1, 1)).flatten()


def get_label_from_filename(filename):
    pattern = 'v_([a-zA-Z]*)_'
    match = re.search(pattern, filename).groups()[0]
    return match


def get_data_pathes(root_path):
    train, val = [], []
    clas = sorted(glob(os.path.join(root_path, '*')))
    for c in clas:
        vids = glob(os.path.join(c, '*'))
        vids = shuffle(vids)
        train += vids[:int(con.split * len(vids))]
        val += vids[int(con.split * len(vids)):]

    return train, val

# if __name__ == '__main__':
# print(get_label_from_filename('v_BenchPress_g01_c01.avi'))
