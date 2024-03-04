import torch
import torch.nn as nn

class GATModel(nn.Module):
    def __init__(self, input_dim=1447,hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        # if base_model == "GRU":
        #     self.rnn = nn.GRU(
        #         input_size=d_feat,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         batch_first=True,
        #         dropout=dropout,
        #     )
        # elif base_model == "LSTM":
        #     self.rnn = nn.LSTM(
        #         input_size=d_feat,
        #         hidden_size=hidden_size,
        #         num_layers=num_layers,
        #         batch_first=True,
        #         dropout=dropout,
        #     )
        # else:
        #     raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.fullyConnect = nn.Linear(input_dim, self.hidden_size)
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1)) # 128 x 1
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x):
        x = self.transformation(x)
        # x: 277 x 64 y: 277 x 64
        sample_num = x.shape[0] 
        dim = x.shape[1] 
        e_x = x.expand(sample_num, sample_num, dim) # 277 x 277 x 64
        e_y = torch.transpose(e_x, 0, 1) # transpose: 277 x 277 x 64 
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2) # 277 x 277 x 128 -> 76729 x 128
        self.a_t = torch.t(self.a) # 1 x 128
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num) # 277 x 277
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight 

    def forward(self, x):
        # out, _ = self.rnn(x) # x : 277 (number of stocks in one day) x 20 (time length) x 20（number of factors） out: 277 x 20 x 64
        # hidden = out[:, -1, :] # 277 x 64 
        hidden = self.fullyConnect(x) # 2171 x 1447 -> 2171 x 64
        hidden = self.leaky_relu(hidden)
        att_weight = self.cal_attention(hidden)  # self attention
        hidden = att_weight.mm(hidden) + hidden 
        hidden = self.fc(hidden) 
        hidden = self.leaky_relu(hidden)
        ans = self.fc_out(hidden)
        ans = ans.squeeze()
        return ans
