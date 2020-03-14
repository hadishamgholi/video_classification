import cv2
import torch
from torch.utils.data import Dataset

import utils


class MyDataset(Dataset):
    def __init__(self, videos_path, t=16):
        self.videos_list = videos_path
        self.t = t

    def __len__(self):
        return len(self.videos_list)

    def __getitem__(self, item):
        item_path = self.videos_list[item]
        cap = cv2.VideoCapture(item_path)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_indices = [int(x * (nframes / self.t)) for x in range(self.t)]
        x = []
        for id in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, id)
            ret, frame = cap.read()
            x.append(torch.FloatTensor(frame))
        cap.release()
        x = torch.stack(x, dim=0)
        y = utils.get_label_from_filename(item_path)
        set_trace()
        y = utils.encode_label(y)
        return x, y
