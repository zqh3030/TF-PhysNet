
import torch.nn as nn
import torch.nn.functional as F
class CNNLSTMModel(nn.Module):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        super(CNNLSTMModel, self).__init__()
        self.n_input = 3
        self.n_target = 3
        self.s =1
        self.cnn1 = nn.Conv1d(self.n_input * self.s, n_hidden, kernel_size=3, stride=1, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.cnn2 = nn.Conv1d(n_hidden, n_hidden, kernel_size=3, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.lstm1 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.fc_x = nn.Linear(n_hidden, n_hidden)
        self.output_layer = nn.Linear(n_hidden, self.n_target* self.s)
    def forward(self, x):
        batch_size, seq_len,_ = x.shape
        x = x.reshape(batch_size, seq_len // self.s, self.n_input * self.s)
        x = x.permute(0, 2, 1)
        out = self.cnn1(x)
        out = F.relu(out)
        out = self.pool1(out)
        out = self.cnn2(out)
        out = F.relu(out)
        out = self.pool2(out)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        out = F.relu(self.fc_x(out))
        y = self.output_layer(out)
        y = y.reshape(batch_size, seq_len, self.n_target)
        return y
