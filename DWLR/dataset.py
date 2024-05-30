import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y=None, n_freq=32) -> None:
        super(MyDataset, self).__init__()
        self.x = torch.from_numpy(x).type(torch.float32)
        print('begin fft')
        self.x_a, self.x_p = self.get_freq(n_freq)
        print('complete fft')
        
        if y is None:
            self.y = y
            self.class_count = None
        else:
            self.class_count = self.get_class_count(y)
            self.y = torch.from_numpy(y).type(torch.long)

    def get_freq(self, n_freq):
        x_ft = torch.fft.rfft(self.x.transpose(1, 2), norm='ortho')
        x_a = x_ft.abs()[:, :, :n_freq]
        x_p = x_ft.angle()[:, :, :n_freq]
        return x_a, x_p

    def get_class_count(self, y) -> torch.Tensor:
        class_num = len(set(y))
        class_count = torch.zeros(class_num)
        for c in y:
            class_count[c] += 1

        return class_count 

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        if self.y is None:
            y = torch.tensor(-1)
        else:
            y = self.y[index]

        return self.x[index], self.x_a[index], self.x_p[index], y

