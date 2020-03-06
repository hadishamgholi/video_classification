from glob import glob
import os, cv2, utils, models
import torch
from sklearn.utils import shuffle
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader

# dataset_path = 'raw_data'
# batch_size = 2
# videos_path = glob(os.path.join(dataset_path, '*.avi'))
# split = .8
# shuffled_dataset = shuffle(videos_path)
# train_videos_path = shuffled_dataset[:int(split * len(videos_path))]
# val_videos_path = shuffled_dataset[int(split * len(videos_path)):]
#
# train_loader = DataLoader(MyDataset(train_videos_path), batch_size=batch_size, shuffle=True, num_workers=6)
# test_loader = DataLoader(MyDataset(val_videos_path), batch_size=batch_size, shuffle=True, num_workers=6)

model = models.C3D()
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

# def train():
#     for i, b in enumerate(train_loader):
#         x, y = b
#         out = model(x)
#         print(out.shape)
#         break


if __name__ == '__main__':
    # train()
    temp = torch.rand((1, 16, 240, 320, 3))
    out = model(temp)
    print(out.shape)