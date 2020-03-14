import re, os
from glob import glob
import config as con
from sklearn.utils import shuffle

def get_label_from_filename(filename):
    pattern = 'v_([a-zA-Z]*)_'
    match = re.search(pattern, filename).groups()[0]
    return match

def get_data_pathes(root_path):
    train, val = [], []
    clas = glob(os.path.join(root_path, '*'))
    for c in clas:
        vids = glob(os.path.join(c, '*'))
        vids = shuffle(vids)
        train += vids[:int(con.split * len(vids))]
        val += vids[int(con.split * len(vids)):]
    
    return train, val

# if __name__ == '__main__':
# print(get_label_from_filename('v_BenchPress_g01_c01.avi'))
