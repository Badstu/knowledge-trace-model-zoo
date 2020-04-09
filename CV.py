from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import random
import pickle

from data.KTData import KTData
from model.RNN import RNN_DKT
from model.CNN import CNN, CNN_3D
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



def init_loss_file(opt):
    # delete loss file while exist
    if os.path.exists(opt.train_loss_path):
        os.remove(opt.train_loss_path)
    if os.path.exists(opt.val_loss_path):
        os.remove(opt.val_loss_path)
    if os.path.exists(opt.test_loss_path):
        os.remove(opt.test_loss_path)


def run_train_valid(opt, vis):
    print(opt.__dict__)
    train_path, valid_path, test_path = init_file_path(opt)

    train_dataset = KTData(train_path, opt='None')
    valid_dataset = KTData(valid_path, opt='None')

    print(train_path, valid_path)
    print(len(train_dataset), len(valid_dataset))

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                             drop_last=True, collate_fn=myutils.collate_fn)


    if opt.model_name == "CNN":
        model = CNN(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    elif opt.model_name == "CNN_3D":
        model = CNN_3D(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)
    else:
        model = RNN_DKT(opt.input_dim, opt.embed_dim, opt.hidden_dim, opt.num_layers, opt.output_dim, opt.batch_size, opt.device)

    lr = opt.lr
    last_epoch = -1
    previous_loss = 1e10

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.99)
    )
    if opt.model_path:
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        model.load_state_dict(checkpoint["model"])
        last_epoch = checkpoint["epoch"]
        lr = checkpoint["lr"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    model = model.to(opt.device)

    train_loss_list = []
    train_auc_list = []
    valid_loss_list = []
    valid_auc_list = []
    # START TRAIN
    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue

        train_loss_meter, train_auc_meter, _ = train.train_3d(opt, vis, model, train_loader, epoch, lr, optimizer)
        val_loss_meter, val_auc_meter, _ = train.valid_3d(opt, vis, model, valid_loader, epoch)

        print("epoch: {}, train_auc: {}, val_auc: {}".format(epoch, train_auc_meter.value()[0], val_auc_meter.value()[0]))

        train_loss_list.append(train_loss_meter.value()[0])
        train_auc_list.append(train_auc_meter.value()[0])

        valid_loss_list.append(val_loss_meter.value()[0])
        valid_auc_list.append(val_auc_meter.value()[0])


        # TODO 每save_every个epoch结束后保存模型参数+optimizer参数
        if epoch % opt.save_every == 0:
            myutils.save_model_weight(opt, model, optimizer, epoch, lr, is_CV=True)

        # TODO 做lr_decay
        lr = myutils.adjust_lr(opt, optimizer, epoch)

    # TODO 结束的时候保存final模型参数
    myutils.save_model_weight(opt, model, optimizer, epoch, lr, is_final=True, is_CV=True)

    return train_loss_list, train_auc_list, valid_loss_list, valid_auc_list


def five_fold_cross_validation(opt, vis):
    rnn_layers_params = [1, 2]
    weight_decay_params = [1e-5, 1e-4, 1e-6, 0]

    loss_result = {} #{("train", 1, 5e-4): matrix[5, 50]}
    auc_result = {}

    for num_layers in rnn_layers_params:
        for weight_decay in weight_decay_params:
            opt.num_layers = num_layers
            opt.weight_decay = weight_decay
            print(num_layers, weight_decay)

            train_loss_matrix = []
            train_auc_matrix = []
            valid_loss_matrix = []
            valid_auc_matrix = []

            for cv_times in range(1, 6):
                opt.cv_times = cv_times
                opt.lr = 0.001
                train_loss_list, train_auc_list, valid_loss_list, valid_auc_list = run_train_valid(opt, vis)

                train_loss_matrix.append(train_loss_list)
                train_auc_matrix.append(train_auc_list)
                valid_loss_matrix.append(valid_loss_list)
                valid_auc_matrix.append(valid_auc_list)

            loss_result[("train", num_layers, weight_decay)] = np.array(train_loss_matrix)
            auc_result[("train", num_layers, weight_decay)] = np.array(train_auc_matrix)
            loss_result[("valid", num_layers, weight_decay)] = np.array(valid_loss_matrix)
            auc_result[("valid", num_layers, weight_decay)] = np.array(valid_auc_matrix)

    return loss_result, auc_result


def main(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    init_loss_file(opt)
    opt.vis = False
    vis = None

    loss_result, auc_result = five_fold_cross_validation(opt, vis)

    with open(opt.save_prefix + 'five_fold_cv.pkl', 'wb') as f:
        pickle.dump({
            "loss_result": loss_result,
            "auc_result": auc_result
        }, f)

    for k, v in loss_result.items():
        loss_matrix = v
        auc_matrix = auc_result[k]

        stage = k[0]
        num_layers = k[1]
        weight_decay = k[2]

        best_loss = loss_matrix.min(axis=1)
        best_auc = auc_matrix.max(axis=1)
        avg_loss = loss_matrix.min(axis=1).mean()
        avg_auc = auc_matrix.max(axis=1).mean()

        print("num_layers:{}, "
              "weight_decay:{}, "
              "stage:{}, "
              "best_loss:{}, "
              "best_auc:{}".format(num_layers,
                                   weight_decay,
                                   stage,
                                   best_loss,
                                   best_auc))
        print("avg_loss:{}, avg_auc:{}".format(avg_loss, avg_auc))


if __name__ == '__main__':
    main(max_epoch=20, model_name="CNN_3D", issave=False)
