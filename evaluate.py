from LSTMModel import Lstm
from dataset import cleanData
from common_parsers import args
import torch


def evlute():
    model = Lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    end_max, end_min, train_loader, test_loader = cleanData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size, seq_len, input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    for i in range(len(preds)):
        print('Predict value is%.2f,True Value is%.2f' % (
        preds[i][0] * (end_max - end_min) + end_min, labels[i] * (end_max - end_min) + end_min))

evlute()