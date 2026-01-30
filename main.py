# -*- coding: utf-8 -*-
import time
import warnings
import numpy as np
import scipy.io
import scipy.io as sio
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

from matplotlib import pyplot as plt

from basecode.loss import Loss_PhyHysLSTM
from basecode.model import CNNLSTMModel
class Dataset:
    def __init__(self, datafile, slice_start, slice_end, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.mat = scipy.io.loadmat(datafile)
        self.X = torch.as_tensor(self.mat['input_ns'][slice_start:slice_end, :])
        self.X = self.X.unsqueeze(-1)
        self.Y_s = torch.as_tensor(self.mat['target_s'][slice_start:slice_end, :])
        self.Y = self.Y_s.unsqueeze(-1)
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
def train_one_epoch1(scaler, Config):
    Config.model_train1.train()
    train_loss = 0

    test_iter = iter(Config.testLoader1)

    for batch, (input_ns, target) in enumerate(Config.trainLoader1):
        pred = Config.model_train1(input_ns)
        loss1 = Config.loss_fn1(input_ns, pred, target)
        loss2 = Config.loss_fn2(pred)

        try:
            input_ns_extra = next(test_iter)[0]
        except StopIteration:
            test_iter = iter(Config.testLoader1)
            input_ns_extra = next(test_iter)[0]

        pred_extra = Config.model_train1(input_ns_extra)
        loss3= Config.loss_fn2(pred_extra)
        total_loss =  1* loss1 + 0.01 * loss2 + 0.01 * loss3
        Config.optimizer1.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(Config.optimizer1)
        scaler.update()
        train_loss += total_loss.item() / len(input_ns)
    return train_loss
def train_loop1(Config):
    Train_loss1 = []
    start_time_all = time.time()
    scaler = GradScaler()
    for epoch in range(Config.max_epoch1):
        Config.model_train1.train()
        start_time = time.time()
        train_loss = train_one_epoch1(scaler, Config)
        Train_loss1.append(train_loss)

        Config.early_stopping1(train_loss, Config.model_train1)
        if Config.early_stopping1.early_stop:
            break
        if epoch % 1 == 0:
            if epoch != 0 and epoch % 20 == 0:
                Test_in_train_Processing(Config=Config, model_test=Config.model_train1)
            time_length = time.time() - start_time
    time_counter(start_time_all, time.time())
    torch.save(
        Config.model_train1,
        Config.model_path1 + 'FModel.pth',
        pickle_protocol=3)
    draw_loss(
        picture_path=Config.picture_path1,
        train_loss=Train_loss1,
        val_loss=Train_loss1,
        plt_save=Config.plt_save,
        plt_show=Config.plt_show)

def Test_in_train_Processing(Config, model_test):
    input_data, test_true, test_pred = test_loop1(Config=Config, model_test=model_test)
    test_s_true, test_s_pred = test_true[...,2], test_pred[...,2]
    f_max(y_true=test_s_true, y_pred=test_s_pred)
    time_performance(label=test_s_true, pred=test_s_pred)
class ModelConfig:
    def get_model(self):
        if self.model_type1 == 'CNNLSTM':
            return CNNLSTMModel(n_hidden=self.N_hidden).to(self.device)
        else:
            raise ValueError(f"error {self.model_type1}")
    def __init__(self, data_name):
        self.data_name = data_name
        self.target_rate = 1
        self.train_num = 50
        self.test_num = 450
        self.train_batch =50
        self.dt = 0.01
        self.model_type1 = 'CNNLSTM'
        self.learning_rate = 5e-3
        self.max_epoch1 = 10000
        self.N_feature = 1
        self.N_hidden = 100
        self.N_target = 1
        self.data_len = 6000
        self.plt_save = True
        self.plt_show = False
        self.BestModel = True
        self.filename1 = '%s_%s_hidden%d_batch%d_lr%0.1e_epoch%d_TgRate%d' % (
            self.model_type1, self.data_name, self.N_hidden, self.train_batch,
            self.learning_rate, self.max_epoch1, self.target_rate)
        print(self.filename1)
        self.data_path = "../data/"
        self.model_path1 = f"results/{self.filename1}/model_save/"
        self.result_path1 = f"results/{self.filename1}/results/"
        self.picture_path1 = f"results/{self.filename1}/pictures/"
        self.testmodel_path1 = self.model_path1 + ('BModel.pth' if self.BestModel else 'FModel.pth')
        print(self.testmodel_path1)

        Folder_checker(path_tuple=(self.data_path, self.model_path1, self.result_path1, self.picture_path1))

        self.trainLoader1, self.testLoader1 = self.get_data_loader()
        self.early_stopping1 = EarlyStopping(patience=2000, verbose=True, path=self.model_path1 + 'BModel.pth')
        self.loss_fn1 = Loss_PhyHysLSTM(dt=self.dt)
        self.FDM = FDM(data_len=self.data_len, dt=self.dt)
        self.model_train1 = self.get_model()
        self.optimizer1 = torch.optim.Adam(self.model_train1.parameters(), self.learning_rate)
    def get_transform(self):
        return Lambda(lambda x_: torch.as_tensor(x_ * self.target_rate).float().cuda())
    def get_data_loader(self):
        def create_data_slice(start, end, batch_size, need_shuffle=True):
            data = Dataset(datafile=self.data_path + self.data_name + ".mat", transform=self.get_transform(),
                           slice_start=start, slice_end=end, target_transform=self.get_transform())
            return DataLoader(data, batch_size=batch_size, shuffle=need_shuffle)

        train_loader = create_data_slice(start=0, end=self.train_num,
                                         batch_size=self.train_batch, need_shuffle=True)

        test_loader = create_data_slice(start=self.train_num,
                                        end=self.train_num + self.test_num,
                                        batch_size=self.train_batch, need_shuffle=False)
        return train_loader, test_loader


def train_and_test_all_models():
    dataname = "real"
    for model_type in ['CNNLSTM', 'DnCNN', 'UNet1D']:
        config = ModelConfig(data_name=dataname)
        config.model_type1 = model_type
        config.filename1 = f"{model_type}_{config.data_name}_hidden{config.N_hidden}_batch{config.train_batch}_lr{config.learning_rate:.0e}_epoch{config.max_epoch1}_TgRate{config.target_rate}"
        config.model_path1 = f"results/{config.filename1}/model_save/"
        config.result_path1 = f"results/{config.filename1}/results/"
        config.picture_path1 = f"results/{config.filename1}/pictures/"
        config.testmodel_path1 = config.model_path1 + ('BModel.pth' if config.BestModel else 'FModel.pth')
        Folder_checker(path_tuple=(config.model_path1, config.result_path1, config.picture_path1))

        config.model_train1 = config.get_model()
        config.optimizer1 = torch.optim.Adam(config.model_train1.parameters(), config.learning_rate)

        train_loop1(config)

def main():
    train_loop1(config)
if __name__ == '__main__':
    dataname = "real"
    config = ModelConfig(data_name=dataname)
    config.model_type1 = 'CNNLSTM'
    config.model_train1 = config.get_model()
    config.optimizer1 = torch.optim.Adam(config.model_train1.parameters(), config.learning_rate)
    main()