from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from dataset import cleanData
from common_parsers import args

def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # Loss funtion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam Gradient Descent

    end_max, end_min, train_loader, test_loader = cleanData(args.corpusFile,args.sequence_length,args.batch_size )
    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)
        if i % 10 == 0:
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('NO%d epoch, model' % i)
    torch.save({'state_dict': model.state_dict()}, args.save_file)

train()