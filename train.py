from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchnet import meter


def train(opt, vis, model, data_loader, epoch, lr, optimizer):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(data_loader)):
        # input: [batch, time_step, input_size](after embedding)
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq) # [batch_size*max_seq_len, 110]

        # TODO mask output to predict
        next_question_number = batch_label[:, :, 0].view(-1).long()
        next_question_label = batch_label[:, :, 1].view(-1)

        mask = torch.zeros_like(output)
        label = []
        for i in range(opt.batch_size):
            start = i * max_seq_len
            len = batch_len[i]
            mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
            label.extend(next_question_label[start:start+len])

        predict = torch.masked_select(output, mask.bool())
        label = torch.Tensor(label).to(opt.device)
        ##############

        loss = criterion(predict, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)
        loss_meter.add(loss.item())

        train_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值
        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch = epoch,
                                                                                                lr = lr,
                                                                                                loss = loss_meter.value()[0],
                                                                                                auc = auc_meter.value()[0]))
    return loss_meter, auc_meter, train_loss_list


def valid(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(valid_loader)):
        max_seq_len = batch_seq.shape[1]
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq)

        # TODO mask output to predict
        next_question_number = batch_label[:, :, 0].view(-1).long()
        next_question_label = batch_label[:, :, 1].view(-1)

        mask = torch.zeros_like(output)
        label = []
        for i in range(opt.batch_size):
            start = i * max_seq_len
            len = batch_len[i]
            mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
            label.extend(next_question_label[start:start+len])

        predict = torch.masked_select(output, mask.bool())
        label = torch.Tensor(label).to(opt.device)
        ##############

        loss = criterion(predict, label)
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)

        auc_meter.add(auc)
        loss_meter.add(loss.item())

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, val_loss_list


def train_3d(opt, vis, model, data_loader, epoch, lr, optimizer):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(data_loader)):
        # input: [batch, time_step, input_size](after embedding)
        max_seq_len = batch_seq.shape[1]
        clips_nums = max_seq_len - opt.k_frames + 1
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq) # [batch_size*clips_nums, 110]

        # TODO 对RNN出来的结果把前k_frames个截断了
        if opt.model_name != "CNN_3D":
            truncat_output = []
            for b in range(opt.batch_size):
                start = b * max_seq_len + opt.k_frames - 1
                end = (b + 1) * max_seq_len
                truncat_output.append(output[start:end, :])

            output = torch.cat([x for x in truncat_output], 0)


        # TODO mask output to predict
        next_question_number = batch_label[:, opt.k_frames:, 0].contiguous().view(-1).long()
        next_question_label = batch_label[:, opt.k_frames:, 1].contiguous().view(-1)

        mask = torch.zeros_like(output)
        label = []
        for i in range(opt.batch_size):
            start = i * clips_nums
            len = batch_len[i] - opt.k_frames + 1
            mask[range(start, start+len), next_question_number[start:start+len] - 1] = True
            label.extend(next_question_label[start:start+len])

        predict = torch.masked_select(output, mask.bool())
        label = torch.Tensor(label).to(opt.device)
        ##############

        loss = criterion(predict, label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        auc = roc_auc_score(label.cpu().data, predict.cpu().data)
        auc_meter.add(auc)
        loss_meter.add(loss.item())

        train_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值
        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch = epoch,
                                                                                                lr = lr,
                                                                                                loss = loss_meter.value()[0],
                                                                                                auc = auc_meter.value()[0]))
    return loss_meter, auc_meter, train_loss_list


def valid_3d(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []
    for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(valid_loader)):
        max_seq_len = batch_seq.shape[1]
        clips_nums = max_seq_len - opt.k_frames + 1
        batch_len = batch_len.to(opt.device)
        batch_seq = batch_seq.to(opt.device)
        batch_label = batch_label.float().to(opt.device)

        output, hidden_state = model(batch_seq)

        # TODO 对RNN出来的结果把前k_frames个截断了
        if opt.model_name != "CNN_3D":
            truncat_output = []
            for b in range(opt.batch_size):
                start = b * max_seq_len + opt.k_frames - 1
                end = (b + 1) * max_seq_len
                truncat_output.append(output[start:end, :])

            output = torch.cat([x for x in truncat_output], 0)

        # TODO mask output to predict
        next_question_number = batch_label[:, opt.k_frames:, 0].contiguous().view(-1).long()
        next_question_label = batch_label[:, opt.k_frames:, 1].contiguous().view(-1)

        mask = torch.zeros_like(output)
        label = []
        for i in range(opt.batch_size):
            start = i * clips_nums
            len = batch_len[i] - opt.k_frames + 1
            mask[range(start, start + len), next_question_number[start:start + len] - 1] = True
            label.extend(next_question_label[start:start + len])

        predict = torch.masked_select(output, mask.bool())
        label = torch.Tensor(label).to(opt.device)
        ##############

        loss = criterion(predict, label)
        auc = roc_auc_score(label.cpu().data, predict.cpu().data)

        auc_meter.add(auc)
        loss_meter.add(loss.item())

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))
    return loss_meter, auc_meter, val_loss_list
