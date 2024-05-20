import os
import copy
from time import time, strftime, localtime
from torch import nn
import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.functional import mse_loss
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error
from egnn import EGNN_Network, TEGN
from utils.utils import parse_args, Logger, set_seed

class predictor(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
        self.out = nn.Sequential(nn.Linear(dim, dim*2), nn.Dropout(p=dropout), nn.GELU(), nn.Linear(dim*2, 3))

    def forward(self, feats, mask=None):
        if exists(mask):
            feats = feats.masked_fill(mask.unsqueeze(-1) == 0, 0)
            return self.out(torch.sum(feats, dim=-2) / torch.sum(mask, dim=-1, keepdim=True))
        return self.out(torch.mean(feats, dim=-2))

def exists(val):
    return val is not None

def run_eval(args, model, loader, y_gt):
    len_loader = len(loader)
    model.eval()
    y_pred = []
    with torch.no_grad():
        batch_ = 0
        for x, pos, y in loader:
            batch_ += 1
            if batch_ != len_loader:
                x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
                mask = (x != 0)
                out = model[0](x, pos, mask=mask)
                out = model[1](out, mask=mask)
                out = out.squeeze()
                y_pred.append(out)
            else:
                if len(x) == 1:
                    y_gt.pop()
                    continue
                else:
                    x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
                    mask = (x != 0)
                    out = model[0](x, pos, mask=mask)
                    out = model[1](out, mask=mask)
                    out = out.squeeze()
                    y_pred.append(out)
    y_pred = torch.cat(y_pred)
    mae = mean_absolute_error(y_gt.numpy(),y_pred.cpu().numpy())

    return mae


def main(number):
    args = parse_args()
    set_seed(args.seed) #可以用seed迭代循环10次，保存在列表里输出AUC-ROC值
    log = Logger(f'{args.save_path}qm9/', f'qm9_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 500
    args.lr = 5e-4 * len(args.gpu.split(','))
    args.bs = 8 * len(args.gpu.split(','))

    # 随机切分切分full.pt,切分索引即可
    x_full_dataset, pos_full_dataset, y_full_dataset = torch.load(f'data_qm9/qm9/full.pt')
    length = len(x_full_dataset)
    indices = list(range(length))
    test_indices_list = []

    test = int(length * 0.1)
    last_test = length - 9 * test
    t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = random_split(indices, lengths=[test, test, test, test, test, test, test, test, test, last_test])
    test_indices_list.append(t1)
    test_indices_list.append(t2)
    test_indices_list.append(t3)
    test_indices_list.append(t4)
    test_indices_list.append(t5)
    test_indices_list.append(t6)
    test_indices_list.append(t7)
    test_indices_list.append(t8)
    test_indices_list.append(t9)
    test_indices_list.append(t10)

    test_indices = test_indices_list[number]
    if number == 9:
        val_number = 0
    else:
        val_number = number + 1
    val_indices = test_indices_list[val_number]

    train_indices = []
    for train_id in test_indices_list:
        if train_id == test_indices or train_id == val_indices:
            continue
        else:
            train_indices.extend(train_id)

    x_train, pos_train, y_train = x_full_dataset[train_indices], pos_full_dataset[train_indices], y_full_dataset[train_indices]
    x_val, pos_val, y_val = x_full_dataset[val_indices], pos_full_dataset[val_indices], y_full_dataset[val_indices]
    x_test, pos_test, y_test = x_full_dataset[test_indices], pos_full_dataset[test_indices], y_full_dataset[test_indices]

    train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, pos_val, y_val), batch_size=args.bs * 2)
    test_loader = DataLoader(TensorDataset(x_test, pos_test, y_test), batch_size=args.bs * 2)
    args.gpu = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # a virtual atom is better than global pooling
    model_1 = TEGN(depth=3, dim=128, pretrain_MTL=True, vocab_size=13).cuda()
    model_2 = predictor(dim=128).cuda()
    model = nn.Sequential(model_1, model_2)
    args.pretrain = 'model_MTL_12_e7.pt'
    if args.pretrain == 'model_MTL_12_e7.pt':
        checkpoint = torch.load(args.save_path + args.pretrain)
        model[0].load_state_dict(checkpoint['model'])
        # for name, param in model[0].named_parameters():
        #     if "encoder_1" in name:
        #         param.requires_grad = False
        #     if "encoder_2" in name:
        #         param.requires_grad = False
        #     if "encoder_3" in name:
        #         param.requires_grad = False
        #     if "encoder_4" in name:
        #         param.requires_grad = False
        #     if "encoder_5" in name:
        #         param.requires_grad = False
        #     if "encoder_6" in name:
        #         param.requires_grad = False
        #     if "encoder_7" in name:
        #         param.requires_grad = False
        #     if "encoder_8" in name:
        #         param.requires_grad = False
        if args.linear_probe:
            for param in model.parameters():
                param.requires_grad = False
    else:
        args.pretrain = 'no_pre'
    model[0].pretrain_MTL = False

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

    best_metric = 1e9
    criterion = torch.nn.L1Loss()
    # optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98))
    optimizer = opt.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98))

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=20, min_lr=5e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    args.data = 'qm9'
    log.logger.info(f'{"=" * 60} qm9 {"=" * 60}\n'
                    f'Embed_dim: {args.dim}; Train: {len(x_train)}; Val: {len(x_val)}; Test: {len(x_test)}; Pre-train Model: {args.pretrain}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}; Linear-probe: {args.linear_probe}\n{"=" * 60} Start Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    mae_list = []
    len_train = len(train_loader)
    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()
        batch = 0
        for x, pos, y in train_loader:
            batch += 1
            if batch != len_train:
                x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
                mask = (x != 0)
                out = model[0](x, pos, mask=mask)
                out = model[1](out, mask=mask)
                out = out.squeeze()
                loss_batch = criterion(out, y.float())
            else:
                if len(x) == 1:
                    continue
                else:
                    x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
                    mask = (x != 0)
                    out = model[0](x, pos, mask=mask)
                    out = model[1](out, mask=mask)
                    out = out.squeeze()
                    loss_batch = criterion(out, y.float())


            # loss += loss_batch.item()
            loss += loss_batch.item() / (len(x_train) * args.bs)

            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metric = run_eval(args, model, val_loader, y_val)

        metric_test = run_eval(args, model, test_loader, y_test)
        mae_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MAE: {:.5f} | test_MAE: {:.5f} | '
            '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric, metric_test,
                                  optimizer.param_groups[0]['lr'] * 1e5))

        lr_scheduler.step(metric)

        if  args.data == 'qm9' and metric < best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)  # deep copy model
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 100: log.logger.info('Early Stopping!!! No Improvement on Loss for 100 Epochs.'); break

    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    checkpoint = {'epochs': args.epochs}

    best_test_mae = min(mae_list)
    best_epoch = mae_list.index(best_test_mae) + 1

    # auroc, auprc, _ = run_eval(args, best_model, test_loader, y_test)
    if len(args.gpu) > 1:
        checkpoint['model'] = best_model.module.state_dict()
    else:
        checkpoint['model'] = best_model.state_dict()
    if args.linear_probe: args.linear_probe = 'Linear'
    torch.save(checkpoint, args.save_path + f'qm9_{args.pretrain}_{args.linear_probe}.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as qm9_{args.pretrain}_{args.linear_probe}.pt.\n')
    log.logger.info('Best Epoch: {} | Test MAE: {:.5f}'.format(best_epoch, best_test_mae))

    return best_test_mae


if __name__ == '__main__':
    best_list = []
    for zhe in range(10):
        best_list.append(main(zhe))
        print('第{}次测试'.format(zhe + 1))

    print(best_list)
    average = sum(best_list)/len(best_list)
    print(average)
