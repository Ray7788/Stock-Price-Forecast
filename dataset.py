from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from common_parsers import argss

# Clean data, data preprocessing, remove invalid data such as ID, stock code, previous day's closing price, transaction date, etc. that are useless for training.
def cleanData(corpusFile,sequence_length,batchSize):
    stock_data = read_csv(corpusFile)
    stock_data.drop('ts_code', axis=1, inplace=True)  # Delete’stock code‘
    stock_data.drop('id', axis=1, inplace=True)  # Delete’id‘
    stock_data.drop('pre_close', axis=1, inplace=True)  # Delete’pre_close‘
    stock_data.drop('trade_date', axis=1, inplace=True)  # Delete’trade_date‘

    end_max = stock_data['close'].max() #Highest closing price
    end_min = stock_data['close'].min() #Lowest closing price
    df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # min-max
    # Construct X and Y
    # Based on the data of the previous n days, predict the closing price (close) of the next day
    sequence = sequence_length
    X = []
    Y = []
    for i in range(df.shape[0] - sequence):
        X.append(np.array(df.iloc[i:(i + sequence), ].values, dtype=np.float32))
        Y.append(np.array(df.iloc[(i + sequence), 0], dtype=np.float32))

    # 构建batch
    total_len = len(Y)
    # print(total_len)

    trainx, trainy = X[:int(0.99 * total_len)], Y[:int(0.99 * total_len)]
    testx, testy = X[int(0.99 * total_len):], Y[int(0.99 * total_len):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True)
    return end_max,end_min,train_loader,test_loader



class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)
