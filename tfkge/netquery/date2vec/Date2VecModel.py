import torch
from torch import nn
import torch.nn.functional as F


class Date2VecConvert:
    def __init__(self, model_path="./d2v_model/d2v_70406_1.5945826793847775.pth"):
        self.model = torch.load(model_path, map_location='cpu').eval()
    
    def __call__(self, x):
        with torch.no_grad():

            return self.model.encode(torch.Tensor(x)).squeeze(0).cpu()

class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin",device='cuda:0'):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.device = device

        self.fc1 = nn.Linear(6, k1)
        # self.fc6 = nn.Linear(16,k1)
        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)


        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(k // 2, 6)
        
        self.fc5 = torch.nn.Linear(6, 6)

        self.convert2year = torch.Tensor([1, 12, 12 * 30, 12 * 30 * 24, 12 * 30 * 24 * 60, 12 * 30 * 24 * 60 * 60]).float().to(device)

        self.activation1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc_mid = nn.Linear(k2*2,k2*2)

        self.layernorm = nn.LayerNorm(k2*2)

    def forward(self, x):

        x = x / self.convert2year
        out1 = self.fc1(x)
        out1 = self.activation1(out1)

        out2 = self.d2(self.activation(self.fc1(x)))
        out = torch.cat([out1, out2], 1)
        out = self.activation2(out)
        out = self.fc_mid(out)

        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out1 = self.activation1(out1)
        # out1 = torch.sigmoid(out1)

        out2 = self.activation(self.fc1(x))
        out = torch.cat([out1, out2], 1)
        out = self.layernorm(out)

        # out = self.activation2(out)
        out = torch.sigmoid(out)
        out = self.fc_mid(out)

        return out

if __name__ == "__main__":
    # model = Date2Vec()
    # inp = torch.randn(1, 6)
    # inp = torch.Tensor([2001,10,12,0,0,0])
    #
    # out = model(inp)
    # print(out)
    # print(out.shape)
    x = torch.Tensor([[[2019, 7, 23, 13, 23, 30]],[[2019, 7, 23, 13, 23, 30]]]).float()
    # x = x.reshape(-1, 1, 6)

    # model = Date2VecConvert('./date2vec/models/d2v_cos/d2v_0_377576.5.pth')
    # a = model(x)
    # print(a)
    x = [[[2019, 7, 23, 13, 23, 30]],[[2019, 7, 23, 13, 23, 30]]]
    spa = Date2VecConvert(model_path='./models/d2v_cos/d2v_108464_2.080166502839707.pth')
    # spa = TimeDirectEncoder(64,6)
    out = spa(x)
    print(out)


