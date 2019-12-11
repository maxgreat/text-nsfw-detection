import torch
import torch.nn as nn
from transformers import BertModel


class FastTextLSTM(nn.Module):
    def __init__(self, nb_layers=1):
        super(FastTextLSTM, self).__init__()
        if nb_layers == 1:
            self.rnn = nn.LSTM(
                input_size=300, hidden_size=300, num_layers=1, batch_first=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=300,
                hidden_size=300,
                num_layers=nb_layers,
                dropout=0.25,
                batch_first=True,
            )
        self.classif = nn.Sequential(
            nn.ReLU(inplace=True), nn.Linear(300, 1, bias=True)
        )

    def forward(self, x, lengths):
        """
            Receive a padded sequence and lengths
        """
        output, _ = self.rnn(x)
        lasts = torch.stack([output[i][lengths[i] - 1] for i in range(output.size(0))])
        return self.classif(lasts).squeeze()



class FastTextSum(nn.Module):
    def __init__(self):
        super(FastTextSum, self).__init__()
        self.classif = nn.Linear(300, 1, bias=True)

    def forward(self, x, lengths):
        return self.classif(x).squeeze()


class BertPooler(nn.Module):
    def __init__(self, bert_model):
        super(BertPooler, self).__init__()
        self.model = BertModel.from_pretrained(bert_model)
        self.model.pooler.dense = nn.Linear(768, 1)

    def forward(self, x, lengths):
        x = self.model(x)
        return x[1].squeeze()


class BertSumer(nn.Module):
    """
        Sum content over buffer_size and pool the result
    """

    def __init__(self, bert_model, buffer_size=12):
        super(BertSumer, self).__init__()
        self.model = BertModel.from_pretrained(bert_model)
        self.classif = nn.Linear(768, 1)
        self.buffer_size = buffer_size

    def forward(self, x, lengths):
        nb_cut = lengths // self.buffer_size
        res = torch.zeros(x.shape[0], 768).cuda()
        if nb_cut > 0:
            for i in range(nb_cut - 1):
                res += self.model(x[i * self.buffer_size : (i + 1) * self.buffer_size])[
                    1
                ]
        res += self.model(x[(nb_cut * self.buffer_size) :])[1]
        res = self.classif(res)
        return res.squeeze()
