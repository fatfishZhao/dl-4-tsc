# FCN
import torch
import numpy as np
import pandas as pd
import time
from torchvision.models import resnet18, resnet50
from utils.train_util import train
import torch.optim as optim
from  torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from utils.torch_ds import get_train_val_dataset, collate_fn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yaml

class Classifier_Resnet_torch:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.verbose = verbose

    def build_model(self, input_shape, nb_classes):
        model = resnet50(pretrained=False)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, nb_classes)
        model.cuda()
        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        import os
        print(os.getcwd())
        with open('./utils/config.yaml') as f:

            config = yaml.load(f)

        # x_val and y_val are only used to monitor the test loss and NOT for training
        scaler = StandardScaler()
        scaler.fit(x_train.reshape(-1, x_train.shape[2]))
        x_train = np.array([scaler.transform(each_x) for each_x in x_train])
        x_val = np.array([scaler.transform(each_x) for each_x in x_val])
        batch_size = config['batch_size']
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))
        nb_epochs = config['nb_epoch']
        optimizer = optim.SGD(self.model.parameters(), lr=config['opt']['lr'], momentum=config['opt']['momentum'],
                              weight_decay=config['opt']['weight_decay'])
        criterion = CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_scheduler']['step_size'],
                                               gamma=config['lr_scheduler']['gamma'])

        data_set = {}
        data_set['train'], data_set['val'] = get_train_val_dataset(x_train, x_val, y_train, y_val, transforms=None)
        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(data_set['train'], batch_size=mini_batch_size,
                                                          shuffle=True, num_workers=4, collate_fn=collate_fn)
        dataloader['val'] = torch.utils.data.DataLoader(data_set['val'], batch_size=mini_batch_size,
                                                        shuffle=True, num_workers=4, collate_fn=collate_fn)



        start_time = time.time()

        hist = train(model=self.model,
              epoch_num=nb_epochs,
              start_epoch=0,
              optimizer=optimizer,
              criterion=criterion,
              exp_lr_scheduler=exp_lr_scheduler,
              data_set=data_set,
              data_loader=dataloader,
              save_dir=self.output_directory)



        duration = time.time() - start_time

        hist_df = pd.DataFrame(hist)
        hist_df = hist_df.iloc[1:,:]
        hist_df.to_csv(self.output_directory+'/hist.csv')
        hist_best = hist_df[hist_df['val_acc']==hist_df['val_acc'].max()]
        hist_best.columns = ['acc','loss','lr','accuracy','val_loss']
        hist_best.to_csv(self.output_directory+'/df_metrics.csv')



        with open(self.output_directory+'/config.yaml', 'w') as f:
            yaml.dump(config, f)
        import matplotlib.pyplot as plt
        plt.plot(hist_df['loss'], 'g')
        plt.plot(hist_df['val_loss'], 'r')
        plt.gca().legend(('TRAIN', 'VAL'))
        plt.yscale('log')
        plt.savefig(self.output_directory+'/loss.png')



