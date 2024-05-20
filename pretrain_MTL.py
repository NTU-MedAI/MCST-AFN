import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import DataLoader
from metrics import AverageMeter
import argparse
from egnn import EGNN_Network, TEGN
from data.dataset_MTL import Smiles_Bert_Dataset,Pretrain_Collater
import time
from utils.utils import parse_args, Logger, set_seed
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--Smiles_head', nargs='+', default=["smiles"], type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 13
dropout_rate = 0.1
args = parse_args()
args.bs = 8
set_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

checkpoint_1 = torch.load(args.save_path + 'pre_n50_5_p1_e1.pt')
checkpoint_2 = torch.load(args.save_path + 'pre_n50_5_p2_e1.pt')
checkpoint_3 = torch.load(args.save_path + 'pre_n50_5_p3_e1.pt')
checkpoint_4 = torch.load(args.save_path + 'pre_n50_5_p4_e1.pt')
checkpoint_5 = torch.load(args.save_path + 'pre_n50_5_p5_e1.pt')
checkpoint_6 = torch.load(args.save_path + 'pre_n50_5_p6_e1.pt')
checkpoint_7 = torch.load(args.save_path + 'pre_n50_5_p7_e1.pt')
checkpoint_8 = torch.load(args.save_path + 'pre_n50_5_p8_e1.pt')

model = TEGN(depth=3, dim=128, pretrain_MTL=True, vocab_size=13).cuda()

model.encoder_1.load_state_dict(checkpoint_1['model'])
model.encoder_2.load_state_dict(checkpoint_2['model'])
model.encoder_3.load_state_dict(checkpoint_3['model'])
model.encoder_4.load_state_dict(checkpoint_4['model'])
model.encoder_5.load_state_dict(checkpoint_5['model'])
model.encoder_6.load_state_dict(checkpoint_6['model'])
model.encoder_7.load_state_dict(checkpoint_7['model'])
model.encoder_8.load_state_dict(checkpoint_8['model'])

if len(args.gpu) > 1:
    model = torch.nn.DataParallel(model)

full_dataset = torch.load('full_dataset.pt')

print(len(full_dataset))
full_dataset_length = int(0.95 * len(full_dataset))
train_size = (full_dataset_length // 19) * 19
test_size = len(full_dataset) - train_size
print(train_size)
print(test_size)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset,batch_size=args.bs,shuffle=True,collate_fn=Pretrain_Collater())
test_dataloader = DataLoader(test_dataset,batch_size=args.bs,shuffle=False,collate_fn=Pretrain_Collater())

optimizer = optim.Adam(model.parameters(),1e-4,betas=(0.9,0.98))

loss_func = nn.CrossEntropyLoss(ignore_index=0,reduction='none')

train_loss = AverageMeter()
train_acc = AverageMeter()
test_loss = AverageMeter()
test_acc = AverageMeter()

def train_step(model, x, pos, y, weights):
    model.train()
    optimizer.zero_grad()
    mask = (x != 0)
    predictions = model(x,pos,mask=mask)
    # print(predictions.shape)
    loss = (loss_func(predictions.transpose(1,2),y)*weights).sum()/weights.sum()
    loss.backward()
    optimizer.step()

    train_loss.update(loss.detach().item(),x.shape[0])
    train_acc.update(((y==predictions.argmax(-1))*weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                     weights.cpu().sum().item())
    try:
        del predictions, loss
    except UnboundLocalError:
        pass
    torch.cuda.empty_cache()


def test_step(model, x, pos, y, weights):
    model.eval()
    mask = (x != 0)
    with torch.no_grad():
        predictions = model(x,pos,mask=mask)
        # print(predictions.shape)
        loss_test = (loss_func(predictions.transpose(1, 2), y) * weights).sum()/weights.sum()

        test_loss.update(loss_test.detach(), x.shape[0])
        test_acc.update(((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item()/weights.cpu().sum().item(),
                              weights.cpu().sum().item())
    try:
        del predictions, loss_test
    except UnboundLocalError:
        pass
    torch.cuda.empty_cache()

for epoch in range(15):
    start = time.time()

    for (batch, (x, pos, y, weights)) in enumerate(train_dataloader):
        train_step(model, x, pos, y, weights)

        if batch%100==0 and batch!=0:
            print('Epoch {} Batch {} training Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.avg))
            print('training Accuracy: {:.4f}'.format(train_acc.avg))

            train_acc.reset()
            train_loss.reset()

    for x, pos, y, weights in test_dataloader:
        test_step(model, x, pos, y, weights)


    print('Epoch {} is Done!'.format(epoch+1))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
    print('training Accuracy: {:.4f}'.format(train_acc.avg))
    print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
    print('test Accuracy: {:.4f}'.format(test_acc.avg))
    checkpoint = {'model': model.state_dict(), 'epochs': epoch}
    test_acc.reset()
    test_loss.reset()

    #这里要修改代码，每次跑完1epoch都要保存一次
    if epoch + 1 == 5:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e5.pt')
    if epoch + 1 == 6:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e6.pt')
    if epoch + 1 == 7:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e7.pt')
    if epoch + 1 == 8:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e8.pt')
    if epoch + 1 == 9:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e9.pt')
    if epoch + 1 == 10:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e10.pt')
    if epoch + 1 == 11:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e11.pt')
    if epoch + 1 == 12:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e12.pt')
    if epoch + 1 == 13:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e13.pt')
    if epoch + 1 == 14:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e14.pt')
    if epoch + 1 == 15:
        torch.save(checkpoint, args.save_path + f'/model_MTL_15_e15.pt')
    # torch.save(model.state_dict(),'weights/' + arch['path']+'_bert_weights{}_{}.pt'.format(arch['name'],epoch+1) )
    # torch.save(model.encoder.state_dict(), 'weights/' + arch['path'] + '_bert_encoder_weights{}_{}.pt'.format(arch['name'], epoch + 1))
    print('Successfully saving checkpoint!!!')