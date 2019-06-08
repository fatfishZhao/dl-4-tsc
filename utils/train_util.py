#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def val(model, data_loader, criterion, data_set):
    # val phase
    model.train(False)  # Set model to evaluate mode

    val_loss = 0
    val_corrects = 0

    for batch_cnt_val, data_val in enumerate(data_loader['val']):
        # print data
        inputs, labels = data_val

        inputs = Variable(inputs.cuda())
        labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

        # forward
        outputs = model(inputs)
        if isinstance(outputs, list):
            loss = criterion(outputs[0], labels)
            loss += criterion(outputs[1], labels)
            outputs = outputs[0]
        else:
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # statistics
        val_loss += loss.data.cpu().numpy()
        batch_corrects = (preds == labels).data.sum()
        val_corrects += batch_corrects

    val_loss = val_loss/len(data_set['val'])
    val_acc = val_corrects.data.cpu().numpy() / len(data_set['val'])
    logging.info('--' * 30)
    return val_acc, val_loss

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir):

    hist={'loss':[0], 'acc':[0], 'lr':[0], 'val_loss':[0], 'val_acc':[0]}

    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode
        all_loss = 0
        all_correct=0

        for batch_cnt, data in enumerate(data_loader['train']):

            model.train(True)
            # print data
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            if isinstance(outputs, list):
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels)
                outputs=outputs[0]
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # batch loss
            _, preds = torch.max(outputs, 1)

            batch_corrects = (preds == labels).data.sum()
            all_correct += batch_corrects
            all_loss += loss.data.cpu().numpy()


        acc = all_correct.data.cpu().numpy()/len(data_set['train'])
        each_train_loss = all_loss/len(data_set['train'])
        print('%d epoch one epoch training finished, acc= %.4f, loss= %.4f, lr= %.4f'%(epoch, acc, each_train_loss, exp_lr_scheduler.get_lr()[0]))
        val_acc, val_loss = val(model, data_loader, criterion, data_set)
        print('val finished, acc= %.4f, loss= %.4f' % (val_acc, val_loss))
        # save model depend on whether it is the best model
        if val_acc>max(hist['val_acc']):
            torch.save(model.state_dict(), save_dir+'best_model.pth')
        hist['loss'].append(each_train_loss)
        hist['acc'].append(acc)
        hist['lr'].append(exp_lr_scheduler.get_lr()[0])
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)


    return hist
