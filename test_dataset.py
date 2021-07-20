# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2020/1/14

import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
from apex import amp
from ignite.engine import create_supervised_evaluator, Events
from modified_ignite_engine import create_supervised_trainer
from IQAdataset import IQADataset
from torch.utils.data import DataLoader
from IQAmodel import IQAModel
from IQAloss import IQALoss
from IQAperformance import IQAPerformance
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import random
from argparse import ArgumentParser


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.architecture, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    test_dataset = IQADataset(args, 'test')
    test_loader = DataLoader(test_dataset)

    # optimizer = Adam([{'params': model.regression.parameters()}, 
    #                   {'params': model.dr6.parameters()},
    #                   {'params': model.dr7.parameters()},
    #                   {'params': model.regr6.parameters()},
    #                   {'params': model.regr7.parameters()},
    #                   {'params': model.features.parameters(), 'lr': 0.0001 * 0.1}],
    #                  lr=0.0001, weight_decay=0) 

    # # Initialization
    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    mapping = True #  args.loss_type != 'mae' and args.loss_type != 'mse'

    checkpoint = torch.load(args.trained_model_file)
    model.load_state_dict(checkpoint['model'])
    k = checkpoint['k']
    b = checkpoint['b']

    evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': 
        IQAPerformance(status='test', k=k, b=b, mapping=mapping)}, device=device)
    evaluator.run(test_loader)
    performance = evaluator.state.metrics
    # TODO: PLCC, RMSE after nonlinear mapping when conducting cross-dataset evaluation
    metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']
    for metric_print in metrics_printed:
        print('{}, {}: {:.3f}'.format(args.dataset, metric_print, performance[metric_print].item()))
    np.save(args.save_result_file, performance)
     
if __name__ == "__main__":
    parser = ArgumentParser(description='Test the Performance of LinearityIQA on a Dataset')
    parser.add_argument("--seed", type=int, default=19920517)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('-flr', '--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')

    parser.add_argument('-arch', '--architecture', default='resnext101_32x8d', type=str,
                        help='arch name (default: resnext101_32x8d)')
    parser.add_argument('-pl', '--pool', default='avg', type=str,
                        help='pool method (default: avg)')
    parser.add_argument('-ubne', '--use_bn_end', action='store_true',
                        help='Use bn at the end of the output?')
    parser.add_argument('-P6', '--P6', type=int, default=1,
                        help='P6 (default: 1)')
    parser.add_argument('-P7', '--P7', type=int, default=1,
                        help='P7 (default: 1)')
    parser.add_argument('-lt', '--loss_type', default='norm-in-norm', type=str,
                        help='loss type (default: norm-in-norm)')
    parser.add_argument('-p', '--p', type=float, default=1,
                        help='p (default: 1)')
    parser.add_argument('-q', '--q', type=float, default=2,
                        help='q (default: 2)')
    parser.add_argument('-a', '--alpha', nargs=2, type=float, default=[1, 0],
                        help='loss coefficient alpha in total loss (default: [1, 0])')
    parser.add_argument('-b', '--beta', nargs=3, type=float, default=[.1, .1, 1],
                        help='loss coefficients for level 6, 7, and 6+7 (default: [.1, .1, 1])')

    parser.add_argument('-modelfile', '--trained_model_file', default=None, type=str,
                        help='trained_model_file')

    parser.add_argument('-ds', '--dataset', default='KonIQ-10k', type=str,
                        help='dataset name (default: KonIQ-10k)')
    parser.add_argument('-eid', '--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0,
                        help='train ratio (default: 0)')
    parser.add_argument('-tvr', '--train_and_val_ratio', type=float, default=0,
                        help='train_and_val_ratio (default: 0)')

    parser.add_argument('-rs', '--resize', action='store_true',
                        help='Resize?')
    parser.add_argument('-rs_h', '--resize_size_h', default=498, type=int,
                        help='resize_size_h (default: 498)')
    parser.add_argument('-rs_w', '--resize_size_w', default=664, type=int,
                        help='resize_size_w (default: 664)')

    parser.add_argument('-augment', '--augmentation', action='store_true',
                        help='Data augmentation?')
    parser.add_argument('-ag', '--angle', default=2, type=float,
                        help='angle (default: 2)')
    parser.add_argument('-cs_h', '--crop_size_h', default=498, type=int,
                        help='crop_size_h (default: 498)')
    parser.add_argument('-cs_w', '--crop_size_w', default=498, type=int,
                        help='crop_size_w (default: 498)')
    parser.add_argument('-hp', '--hflip_p', default=0.5, type=float,
                        help='hfilp_p (default: 0.5)')
    
    args = parser.parse_args()

    # KonIQ-10k that train-val-test split provided by the owner. 
    # The model is trained on KonIQ-10k train set,
    # and the best model is selected based on the KonIQ-10k val set.
    # Only the KonIQ-10k test set will be tested.
    # If you train a model on other dataset and want to test the whole KonIQ-10k dataset,
    # you should modify args.train_ratio and args.train_and_val_ratio to 0.
    if args.dataset == 'KonIQ-10k':
        args.train_ratio = 7058/10073
        args.train_and_val_ratio = 8058/10073
        if not args.resize:
            args.resize_size_h = 768
            args.resize_size_w = 1024

    args.im_dirs = {'KonIQ-10k': 'KonIQ-10k',  
                    'CLIVE': 'CLIVE' 
                    }  # ln -s database_path xxx
    args.data_info = {'KonIQ-10k': './data/KonIQ-10kinfo.mat',
                      'CLIVE': './data/CLIVEinfo.mat'}

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.trained_model_file is None:
        args.format_str = '{}-{}-bn_end={}-loss={}-p={}-q={}-detach-False-ft_lr_ratio={}-alpha={}-beta={}-KonIQ-10k-res={}-{}x{}-aug={}-monotonicity=False-lr={}-bs={}-e={}-opt_level=O1-EXP{}'\
                      .format(args.architecture, args.pool, args.use_bn_end, args.loss_type, args.p, args.q, args.ft_lr_ratio, args.alpha, args.beta, 
                              args.resize, args.resize_size_h, args.resize_size_w, args.augmentation, args.learning_rate, args.batch_size, args.epochs, args.exp_id)
        args.trained_model_file = 'checkpoints/' + args.format_str

    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/dataset={}-tested_on_{}'.format(args.dataset, os.path.split(args.trained_model_file)[1])
    print(args)
    run(args)
