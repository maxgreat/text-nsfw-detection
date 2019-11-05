import os

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

import fastText


def collate(data):
    captions, labels = zip(*data)
    lengths = [len(cap) for cap in captions]
    targets = pad_sequence(captions, batch_first=True)
    return targets, lengths, torch.Tensor(labels)


def collate_simple(data):
    captions, labels = zip(*data)
    lengths = [1 for i in range(len(captions))]
    return torch.stack(captions), lengths, torch.Tensor(labels)


class GenericDataset(data.Dataset):
    """
        Dataset for pornographique website
    """

    def __init__(self, root, sset, tokenizer, getter):
        self.root = os.path.join(root, sset + ".txt")
        self.data = [tokenizer(line) for line in open(self.root)]
        self.getter = getter

    def __getitem__(self, index, raw=False):
        if raw:
            return self.getter(self.data), self.data[index]
        else:
            return self.getter(self.data[index])

    def __len__(self):
        return len(self.data)


class FastTextDataset(GenericDataset):
    def __init__(self, file, sum=False):
        self.data = []
        for line in open(file):
            label, sentence = line.split(" ", 1)
            if label == "__label__pron" or label == "__label__1":
                label = 1
            else:
                label = -1
            self.data.append((sentence.split(" "), label))
        fasttext_model = fastText.load_model("/data/m.portaz/cc.fr.300.bin")
        if sum:
            self.getter = lambda x: (
                torch.sum(
                    torch.Tensor([fasttext_model.get_word_vector(w) for w in x[0]]), 0
                ),
                x[1],
            )
        else:
            self.getter = lambda x: (
                torch.Tensor([fasttext_model.get_word_vector(w) for w in x[0]]),
                x[1],
            )


class BertTextDataset(GenericDataset):
    def __init__(self, file, bert_model):
        self.data = []
        for line in open(file):
            label, sentence = line.split(" ", 1)
            if len(sentence.split(" ")) > 20:
                continue
            if (
                label == "__label__pron"
                or label == "__label__1"
                or label == "__label__porn"
            ):
                label = 1
            else:
                label = -1
            self.data.append((sentence, label))
        berttok = BertTokenizer.from_pretrained(bert_model)
        self.getter = lambda x: (torch.LongTensor(berttok.encode(x[0])), x[1])
