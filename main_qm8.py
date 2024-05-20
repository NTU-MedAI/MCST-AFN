import os
import copy
from time import time, strftime, localtime
from torch import nn
import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from scipy import stats
from sklearn.metrics import roc_auc_score,mean_absolute_error
from egnn import EGNN_Network, TEGN
from utils.utils import parse_args, Logger, set_seed


class predictor(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout
        self.out = nn.Sequential(nn.Linear(dim, dim*2), nn.Dropout(p=dropout), nn.GELU(), nn.Linear(dim*2, 12))

    def forward(self, feats, mask=None):
        if exists(mask):
            feats = feats.masked_fill(mask.unsqueeze(-1) == 0, 0)
            return self.out(torch.sum(feats, dim=-2) / torch.sum(mask, dim=-1, keepdim=True))
        return self.out(torch.mean(feats, dim=-2))

def exists(val):
    return val is not None

def run_eval(args, model, loader, y_gt):
    model.eval()
    # metric = 0
    y_pred = []
    with torch.no_grad():
        for x, pos, y in loader:
            x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
            mask = (x != 0)
            # out = model(x, pos, mask=mask)[1][..., 0]
            # out = torch.sigmoid(out)
            out = model[0](x, pos, mask=mask)
            out = model[1](out, mask=mask)
            out = out.squeeze()
            y_pred.append(out)
            # metric += nn.L1loss(out, y, reduction='sum').item() / len(y_gt)
    # y_gt = y_gt.view(-1)
    y_pred = torch.cat(y_pred)
    # y_pred = y_pred.view(-1)
    # spearman = stats.spearmanr(y_pred.cpu().numpy(), y_gt.numpy())[0]
    # pearson = stats.pearsonr(y_pred.cpu().numpy(), y_gt.numpy())[0]
    mae = mean_absolute_error(y_gt.numpy(),y_pred.cpu().numpy())
    return mae


def main():
    args = parse_args()
    set_seed(args.seed) #可以用seed迭代循环10次，保存在列表里输出AUC-ROC值
    log = Logger(f'{args.save_path}qm8/', f'qm8_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    args.epochs = 500
    args.lr = 5e-4 * len(args.gpu.split(','))
    args.bs = 8 * len(args.gpu.split(','))
    args.data = 'qm8'

    x_train, pos_train, y_train = torch.load(f'data_qm8/qm8/train.pt')
    x_val, pos_val, y_val = torch.load(f'data_qm8/qm8/val.pt')
    x_test, pos_test, y_test = torch.load(f'data_qm8/qm8/test.pt')

    train_loader = DataLoader(TensorDataset(x_train, pos_train, y_train), batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, pos_val, y_val), batch_size=args.bs)
    test_loader = DataLoader(TensorDataset(x_test, pos_test, y_test), batch_size=args.bs)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # a virtual atom is better than global pooling
    model_1 = TEGN(depth=3, dim=128, pretrain_MTL=True, vocab_size=13).cuda()
    model_2 = predictor(dim=128).cuda()
    model = nn.Sequential(model_1, model_2)
    args.pretrain = 'model_MTL_15_e15.pt'

    if args.pretrain == 'model_MTL_15_e15.pt':
        checkpoint = torch.load(args.save_path + args.pretrain)
        model[0].load_state_dict(checkpoint['model'])
        for name, param in model[0].named_parameters():
            if "encoder_1" in name:
                param.requires_grad = False
            if "encoder_2" in name:
                param.requires_grad = False
            if "encoder_3" in name:
                param.requires_grad = False
            if "encoder_4" in name:
                param.requires_grad = False
            if "encoder_5" in name:
                param.requires_grad = False
            if "encoder_6" in name:
                param.requires_grad = False
            if "encoder_7" in name:
                param.requires_grad = False
            if "encoder_8" in name:
                param.requires_grad = False
        if args.linear_probe:
            for param in model[0].parameters():
                param.requires_grad = False
    else:
        args.pretrain = 'no_pre'
    model[0].pretrain_MTL = False

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)

    best_metric = 1e9
    criterion = torch.nn.L1Loss()
    # optimizer = opt.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98))
    optimizer = opt.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98))

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    log.logger.info(f'{"=" * 60} qm8 {"=" * 60}\n'
                    f'Embed_dim: {args.dim}; Train: {len(x_train)}; Val: {len(x_val)}; Test: {len(x_test)}; Pre-train Model: {args.pretrain}'
                    f'\nTarget: {args.data}; Batch_size: {args.bs}; Linear-probe: {args.linear_probe}\n{"=" * 60} Start Training {"=" * 60}')

    t0 = time()
    early_stop = 0
    mae_list = []
    # try:
    for epoch in range(0, args.epochs):
        model.train()
        loss = 0.0
        t1 = time()
        for x, pos, y in train_loader:
            x, pos, y = x.long().cuda(), pos.float().cuda(), y.cuda()
            mask = (x != 0)
            out = model[0](x, pos, mask=mask)
            out = model[1](out,mask = mask)
            out = out.squeeze()
            loss_batch = criterion(out, y.float())
            loss += loss_batch.item()/(len(x_train) * args.bs)

            scaler.scale(loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        metric = run_eval(args, model, val_loader, y_val)
        metric = metric

        metric_test = run_eval(args, model, test_loader, y_test)
        mae_list.append(metric_test)

        log.logger.info(
            'Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MAE: {:.5f} | test_MAE: {:.5f} | '
                        '| Lr: {:.3f}'.format(epoch + 1, time() - t1, loss * 1e4, metric , metric_test,
                                              optimizer.param_groups[0]['lr'] * 1e5))

        lr_scheduler.step(metric)

        if  args.data == 'qm8' and metric < best_metric:
            best_metric = metric
            best_model = copy.deepcopy(model)  # deep copy model
            best_epoch = epoch + 1
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 50: log.logger.info('Early Stopping!!! No Improvement on Loss for 50 Epochs.'); break
    # except:
    #     log.logger.info('Training is interrupted.')
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
    torch.save(checkpoint, args.save_path + f'qm8_{args.pretrain}_{args.linear_probe}.pt')
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 20, (time() - t0) / 3600, "=" * 20))
    log.logger.info(f'Save the best model as qm8_{args.pretrain}_{args.linear_probe}.pt.\n')
    log.logger.info('Best Epoch: {} | Test MAE: {:.5f}'.format(best_epoch, best_test_mae))

if __name__ == '__main__':
    main()