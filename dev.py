from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import random
# from data.assit2009 import DataAssist2009
from data.KTData import KTData
from model.RNN import RNN_DKT
from model.CNN import CNN
from model.CNN import CNN_3D

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import myutils
from utils.visualize import Visualizer
from utils.config import Config
from utils.file_path import init_file_path
from tqdm import tqdm
from torchnet import meter

import train
import test




def run_model(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.model_name == "CNN":
        model = CNN(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size,
                    opt.device)
    elif opt.model_name == "CNN_3D":
        model = CNN_3D(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size,
                    opt.device)
    else:
        model = RNN_DKT(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size,
                        opt.device)

    test_input = torch.randn((4, 180))


def main():
    input_dim = 220
    embed_dim = 200
    hidden_dim = 225
    output_dim = 110
    batch_size = 4
    device = torch.device('cpu')

    test_input = torch.randint(1, 110, (4, 200))

    model = CNN_3D(input_dim, embed_dim, hidden_dim, 1, output_dim, batch_size, device)
    out = model(test_input)

    train_path = "D:\\workspaces\\CKT\\dataset\\assist2009_updated\\sayhi_test.csv"
    train_dataset = KTData(train_path, opt='None')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=myutils.collate_fn)
    batch_len, batch_seq, batch_label = next(iter(train_loader))

    print(batch_len, batch_seq.shape, batch_label.shape)

    k_frames = 8
    next_question_number = batch_label[:, k_frames:, 0].contiguous().view(-1).long()
    next_question_label = batch_label[:, k_frames:, 1].contiguous().view(-1)

    print(next_question_number.shape)
    print(next_question_label.shape)


if __name__ == '__main__':
    # run_model(model_name = 'CNN',
    #          batch_size = 64,
    #          num_layers=2,
    #          lr=0.001,
    #          lr_decay=0.3,
    #          decay_every_epoch=5,
    #          weight_decay = 5e-4)

    # main(model_name="RNN", lr_decay = 1, weight_decay=0, hidden_dim=200)

    main()
