import argparse
import json
import cv2
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# 將照片壓縮城64*64
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, object_data):
        self.image_paths = image_paths
        self.labels = labels
        self.object_data = object_data
        self.dotransform = Resize((64, 64))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = torch.from_numpy(self.labels[idx])

        # Load and preprocess the image
        image = cv2.imread(image_path)
        # 從BGR變成RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # 變成0～1 nomsrlization
        image = image.astype(np.float32) / 255.0 

        # print(image.shape)
        image = torch.from_numpy(image).permute(2, 0, 1)

        image = self.dotransform(image)
        return image, label


class ResidualConvBlock(nn.Module):
    # 如果沒有指定函數的返回值，則默認返回 None

    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            nn.Tanh(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果output大小改的話conv會出錯（dim不同）
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # 如果過程中使通道數增加，這樣設計才會使殘差是對的
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            # 輸出前做類似正規化
            return out / 1.414 
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


"""
UnetDown will divide the tensor's width and length by two
"""

# downsampling的功能
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        down_layers = nn.Sequential(ResidualConvBlock(in_channels, out_channels), 
                                    nn.MaxPool2d(2),)
        
        self.model = down_layers

    def forward(self, x):
        return self.model(x)


"""
UnetUp will times the tensor's width and length by two
"""


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        # 來實現一個卷積反轉層，作用是將上一層的特徵映射轉換回輸入圖像的大小，並且保留了更多的資訊
        up_layers = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                                    ResidualConvBlock(out_channels, out_channels),
                                    ResidualConvBlock(out_channels, out_channels),)

        self.model = up_layers

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


"""
Embed time information and conditional(you can embed position as well)
"""

# 用來將各個條件(time或文字條件）壓成特徵向量
class EmbedFC(nn.Module): 
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        Embed_layers = nn.Sequential(nn.Linear(input_dim, emb_dim),
                                        nn.Tanh(),
                                        nn.Linear(emb_dim, emb_dim*2),
                                        nn.Tanh(),
                                        nn.Linear(emb_dim*2, emb_dim),)
        self.model = Embed_layers

    def forward(self, x):
        x = x.view(-1, self.input_dim).float()
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_classes=24):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        """
        Be careful with this part, use nn.AvgPool2d() with the number that can divide current width
        and length with no remain
        """
        self.to_vec = nn.Sequential(nn.AvgPool2d(8), nn.Tanh())
        # 不同層的時間特徵
        self.timeembed0 = EmbedFC(1, 4*n_feat)
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        # 不同層的condition
        self.contextembed0 = EmbedFC(n_classes, 4*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        """
        If you change nn.AvgPool2d above, remember to set the last argment in nn.ConvTranspose2d
        carefully, it will upsampling the tensor.
        """
        # 用來將特徵壓更低
        self.bottle_neck = nn.Sequential(
            # 於將輸入圖像的特徵映射到較小的空間中
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, 8, 8),  # otherwise just have 2*n_feat
            # 來實現一個分組的批次正規化（Group Normalization）操作
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )
        self.up0 = UnetUp(8 * n_feat, 2*n_feat)
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    # x是照片, c是文字條件, t 是 timestep
    def forward(self, x, c, t, context_mask=None):
        # down layer
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)
        # 各層將條件與time embeded 進來
        cemb0 = self.contextembed0(c).view(-1, self.n_feat * 4, 1, 1)
        temb0 = self.timeembed0(t).view(-1, self.n_feat * 4, 1, 1)
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)


        # 最終先層bottle_neck, 並在upsampling每層中加入time與condition
        up0 = self.bottle_neck(hiddenvec)
        up1 = self.up0(cemb0 * up0 + temb0, down3)

        # print(cemb1.size())
        # print(up1.size())

        up2 = self.up1(cemb1 * up1 + temb1, down2) 
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

# 用來獲得預先計算的beta和alpha值
def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer 是用來獲得ddpm_schedules中的各種參數
        # self.register_buffer是一种特殊的方法，用于在PyTorch模型的类中注册持久化的缓冲区。
        # 缓冲区是一种在模型中保存持久状态的方式，它不会被视为模型的可训练参数，也不会在反向传播过程中更新
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        # 採樣時間(由1~n_T來隨機生成)
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  
        # noise為标准正态分布（均值为0，标准差为1）
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # x_t 是sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # 代表原始照片經過我們network生成的noise
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        # loss 是我們預測出來的noise和新增進去的noise(guassin noise)，此使用MSE
        loss = self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, c))
        return loss

    def sample(self, n_sample, size, device, guide_w=0.0):
        # 採樣隨機的guassian noise
        x_i = torch.randn(n_sample, *size).to(device)
        # 採樣的條件初始化與輸入 
        c_i = torch.arange(0, 24).to(device)  
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))
        context_mask = torch.zeros_like(c_i).to(device)
        # 讓batch變兩倍
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.  

        # 檢查所要生成的條件
        x_i_store = []  
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # 分割預測跟計算權重
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def test_sample(self, n_sample, size, cond, device, guide_w=0.0):
        """
        n_sample means how many pictures
        size means size for every picture
        cond means condition for every picture, size is n_sample * one_hot(args.n_objects)
        device is either cuda or cpu
        guid_w remain 0 now
        """
        x_i_store = []
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        assert n_sample == len(cond), "n_sample's length should be same as cond"
        for i in range(self.n_T, 0, -1):
            print(f"test_sampling timestep: {i}", end="\r")
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i, cond, t_is, context_mask=None)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="please input batch_size")
    parser.add_argument("--data_train_path", default="./iclevr", help="please input images' location")
    parser.add_argument("--train_path", default="./dataset/train.json", help="path for train json file")
    parser.add_argument("--test_path", default="./dataset/test.json", help="path for test json file")
    parser.add_argument("--new_test_path", default="./dataset/new_test.json", help="path for test json file")
    parser.add_argument("--object_path", default="./dataset/objects.json", help="path for object json file")
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--n_objects", type=int, default=24, help="number of objects")
    parser.add_argument("--n_feats", type=int, default=512, help="number of feats")
    parser.add_argument("--lrate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_T", type=float, default=300, help="n_T")
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    """
    start getting dataset
    """
    with open(args.train_path, 'r') as f:
        train_data = json.load(f)
    with open(args.test_path, 'r') as f:
        test_data = json.load(f)
    with open(args.new_test_path, 'r') as f:
        new_test_data = json.load(f)
    with open(args.object_path, 'r') as f:
        object_data = json.load(f)

    #  get train data & one hot
    train_image_name_list = []
    train_labels_list = []
    test_cond_list = []
    new_test_cond_list = []

    
    for image_name, labels in train_data.items():
        # 修正資料名與生成one hot array
        train_image_name_list.append(args.data_train_path + "/" + image_name)
        one_hot_array = np.zeros((args.n_objects,), dtype=float)


        for label in labels:
            one_hot_array[object_data[label]] = 1
        train_labels_list.append(one_hot_array)

    # 獲得test的one hot 並存入 test_cond_list
    for conds in test_data:

        one_hot_array = np.zeros((args.n_objects,), dtype=float)
        for cond in conds:
            one_hot_array[object_data[cond]] = 1
        test_cond_list.append(one_hot_array)

    #  獲得new test的one hot 並存入 new_test_cond_list
    for conds in new_test_data:
        one_hot_array = np.zeros((args.n_objects,), dtype=float)
        for cond in conds:
            one_hot_array[object_data[cond]] = 1
        new_test_cond_list.append(one_hot_array)

    train_dataset = CustomDataset(train_image_name_list, train_labels_list, object_data)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=args.n_feats, n_classes=args.n_objects), betas=(1e-4, 0.02), n_T=args.n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=args.lrate)
    for ep in range(args.n_epochs):
        # train_dataset = torch.utils.data.Subset(train_dataset, range(50))  # This line is for testing, comment it before training
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{args.n_epochs}")
        loss_ema = None
        for x, label in pbar:
            optim.zero_grad()
            x = x.to(device)
            label = label.to(device)
            ddpm.train()
            loss = ddpm(x, label)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        ddpm.eval()
        if ep > 15:
            with torch.no_grad():
                n_sample = len(test_cond_list)
                cond_tensor = torch.FloatTensor(np.array(test_cond_list)).to(device)
                x_gen, x_gen_store = ddpm.test_sample(n_sample, (3, 64, 64), cond_tensor, device=device)
                save_image(x_gen, f'./output/test/output_epoch{ep}.png', nrow=8, normalize=True, range=(-1, 1))

                n_sample = len(new_test_cond_list)
                cond_tensor = torch.FloatTensor(np.array(new_test_cond_list)).to(device)
                x_gen, x_gen_store = ddpm.test_sample(n_sample, (3, 64, 64), cond_tensor, device=device)
                save_image(x_gen, f'./output/new_test/output_epoch{ep}.png', nrow=8, normalize=True, range=(-1, 1))

        torch.cuda.empty_cache()
