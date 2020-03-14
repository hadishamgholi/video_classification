from glob import glob
import os, cv2, utils, models
import torch
from sklearn.utils import shuffle
from torch import nn, optim
import config as con
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader


train_path , val_path = utils.get_data_pathes(con.dataset_path)

train_loader = DataLoader(MyDataset(train_path), batch_size=con.batch_size, shuffle=True, num_workers=6)
val_loader = DataLoader(MyDataset(val_path), batch_size=con.batch_size, shuffle=True, num_workers=6)

model = models.C3D()
# total_params = sum(p.numel() for p in model.parameters())
# print(total_params)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=.001)


def train():
    epoch_train_loss = 0
    for i, b in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x, y = b
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
    return epoch_train_loss


def validation():
    epoch_val_loss = 0
    for i, b in enumerate(val_loader):
        model.eval()
        x, y = b
        with torch.no_grad():
            out = model(x)
            loss = criterion(out, y)
        epoch_val_loss += loss.item()
    return epoch_val_loss


def run():
    train_losses, val_losses = [], []
    for e in range(con.epochs):
        print(f'training epoch {e} ... ')
        t_loss = train()
        v_loss = validation()
        print(f'validating ...')
        train_losses.append(t_loss)
        val_losses.append(v_loss)


if __name__ == '__main__':
    run()
