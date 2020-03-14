from torch import nn
import torch.nn.functional as F


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        # block one
        seq = []
        seq.append(nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, dilation=1, padding=1))
        for i in [2, 4, 8]:
            seq.append(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, dilation=i, padding=i))
        self.block1 = nn.Sequential(*seq, nn.BatchNorm3d(64), nn.ReLU())

        # to reduce the frame size
        self.reduce_size1 = nn.Conv3d(in_channels=64, out_channels=64, stride=(1, 2, 2), kernel_size=3,
                                      padding=(1, 0, 0))

        # block two
        seq = []
        seq.append(nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, dilation=1, padding=1))
        for i in [2, 4, 8]:
            seq.append(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, dilation=i, padding=i))
        self.block2 = nn.Sequential(*seq, nn.BatchNorm3d(32), nn.ReLU())

        # to reduce the frame size
        self.reduce_size2 = nn.Conv3d(in_channels=32, out_channels=32, stride=2, kernel_size=3)

        self.fc = nn.Sequential(nn.Linear(32 * 3 * 14 * 19, 512),
                                nn.Dropout(.2),
                                nn.Linear(512, 101),
                                nn.Softmax())
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        s = x.shape
        # make channel first
        x = x.reshape(s[0], s[-1], *s[1:4])
        x = self.block1(x)
        x = self.reduce_size1(x)
        x = self.reduce_size1(x)
        x = self.block2(x)
        x = self.reduce_size2(x)
        x = self.reduce_size2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
