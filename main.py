# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2020/1/14

import torch
from torch.optim import Adam, SGD, Adadelta, lr_scheduler
from apex import amp
from ignite.engine import create_supervised_evaluator, Events
from modified_ignite_engine import create_supervised_trainer
from IQAdataset import get_data_loaders
from IQAmodel import IQAModel
from IQAloss import IQALoss
from IQAperformance import IQAPerformance
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import random
from argparse import ArgumentParser


metrics_printed = ['SROCC', 'PLCC', 'RMSE', 'SROCC1', 'PLCC1', 'RMSE1', 'SROCC2', 'PLCC2', 'RMSE2']
def writer_add_scalar(writer, status, dataset, scalars, iter):
    for metric_print in metrics_printed:
        writer.add_scalar('{}/{}/{}'.format(status, dataset, metric_print), scalars[metric_print], iter)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.architecture, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    print(model)
    if args.ft_lr_ratio == .0:
        for param in model.features.parameters():
            param.requires_grad = False
    train_loader, val_loader, test_loader = get_data_loaders(args)

    optimizer = Adam([{'params': model.regression.parameters()}, # The most important parameters. Maybe we need three levels of lrs
                      {'params': model.dr6.parameters()},
                      {'params': model.dr7.parameters()},
                      {'params': model.regr6.parameters()},
                      {'params': model.regr7.parameters()},
                      {'params': model.features.parameters(), 'lr': args.learning_rate * args.ft_lr_ratio}],
                     lr=args.learning_rate, weight_decay=args.weight_decay) # Adam can be changed to other optimizers, such as SGD, Adadelta.

    # Initialization
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    mapping = True #  args.loss_type != 'mae' and args.loss_type != 'mse'

    if args.evaluate:
        checkpoint = torch.load(args.trained_model_file)
        model.load_state_dict(checkpoint['model'])
        k = checkpoint['k']
        b = checkpoint['b']

        evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': 
            IQAPerformance(status='test', k=k, b=b, mapping=mapping)}, device=device)
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        for metric_print in metrics_printed:
            print('{}, {}: {:.3f}'.format(args.dataset, metric_print, performance[metric_print].item()))
        for metric_print in metrics_printed:
            print('{:.3f}'.format(performance[metric_print].item()))
        np.save(args.save_result_file, performance)
        return

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay)
    loss_func = IQALoss(loss_type=args.loss_type, alpha=args.alpha, beta=args.beta, p=args.p, q=args.q, 
                        monotonicity_regularization=args.monotonicity_regularization, gamma=args.gamma, detach=args.detach)
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=device, accumulation_steps=args.accumulation_steps)

    if args.pbar:
        from ignite.contrib.handlers import ProgressBar

        ProgressBar().attach(trainer)

    evaluator_for_train = create_supervised_evaluator(model, metrics={'IQA_performance': 
        IQAPerformance(status='train', mapping=mapping)}, device=device)

    current_time = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    writer = SummaryWriter(log_dir='{}/{}-{}'.format(args.log_dir, args.format_str, current_time))
    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = -100, -1  # larger, better, e.g., SROCC or PLCC. If RMSE is used, best_val_criterion <- 10000

    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        if args.test_during_training:
            evaluator_for_train.run(train_loader) # It is better to re-make a train_loader_for_evaluation so as not to disturb the random number generator.
            performance = evaluator_for_train.state.metrics
            writer_add_scalar(writer, 'train', args.dataset, performance, engine.state.epoch)
            k = performance['k']
            b = performance['b']
        else:
            k = [1, 1, 1]
            b = [0, 0, 0]

        evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': 
            IQAPerformance(status='test', k=k, b=b, mapping=mapping)}, device=device)
        evaluator.run(val_loader)
        performance = evaluator.state.metrics
        writer_add_scalar(writer, 'val', args.dataset, performance, engine.state.epoch)
        val_criterion = abs(performance[args.val_criterion])  # when alpha=[0,1],loss_type='linearity', test_during_training=False, SROCC/PLCC can be negative during training.
        if args.test_during_training:
            evaluator.run(test_loader)
            performance = evaluator.state.metrics
            writer_add_scalar(writer, 'test', args.dataset, performance, engine.state.epoch)

        global best_val_criterion, best_epoch
        if val_criterion > best_val_criterion: # If RMSE is used, then change ">" to "<".
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'k': k,
                'b': b
            }
            torch.save(checkpoint, args.trained_model_file)
            best_val_criterion = val_criterion
            best_epoch = engine.state.epoch
            print('Save current best model @best_val_criterion ({}): {:.3f} @epoch: {}'.format(args.val_criterion, best_val_criterion, best_epoch))
        else:
            print('Model is not updated @val_criterion ({}): {:.3f} @epoch: {}'.format(args.val_criterion, val_criterion, engine.state.epoch))

        scheduler.step(engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        writer.close ()  # close the Tensorboard writer
        print('best epoch: {}'.format(best_epoch))
        checkpoint = torch.load(args.trained_model_file)
        model.load_state_dict(checkpoint['model'])
        if args.test_during_training:
            k = checkpoint['k']
            b = checkpoint['b']
        else:
            evaluator_for_train.run(train_loader)
            performance = evaluator_for_train.state.metrics
            k = performance['k']
            b = performance['b']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'k': k,
                'b': b
            }
            torch.save(checkpoint, args.trained_model_file)

        evaluator = create_supervised_evaluator(model, metrics={'IQA_performance': 
            IQAPerformance(status='test', k=k, b=b, mapping=mapping)}, device=device)
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        for metric_print in metrics_printed:
            print('{}, {}: {:.3f}'.format(args.dataset, metric_print, performance[metric_print].item()))
        for metric_print in metrics_printed:
            print('{:.3f}'.format(performance[metric_print].item()))
        np.save(args.save_result_file, performance)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    parser = ArgumentParser(description='Norm-in-Norm Loss with Faster Convergence and Better Performance for Image Quality Assessment')
    parser.add_argument("--seed", type=int, default=19920517)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=8,
                        help='batch size for training (default: 8)')
    parser.add_argument('-flr', '--ft_lr_ratio', type=float, default=0.1,
                        help='ft_lr_ratio (default: 0.1)')
    parser.add_argument('-accum', '--accumulation_steps', type=int, default=1,
                        help='accumulation_steps for training (default: 1)')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.1,
                        help='lr decay (default: 0.1)')
    parser.add_argument('-olrd', '--overall_lr_decay', type=float, default=0.01,
                        help='overall lr decay (default: 0.01)')
    parser.add_argument('-optl', '--opt_level', default='O1', type=str,
                        help='opt_level for amp (default: O1)')
    parser.add_argument('-rn', '--randomness', action='store_true',
                        help='Allow randomness during training?')
    parser.add_argument('-valc', '--val_criterion', default='SROCC', type=str,
                        help='val_criterion: SROCC or PLCC (default: SROCC)') # If using RMSE, minor modification should be made, i.e., 

    parser.add_argument('-a', '--alpha', nargs=2, type=float, default=[1, 0],
                        help='loss coefficient alpha in total loss (default: [1, 0])')
    parser.add_argument('-b', '--beta', nargs=3, type=float, default=[.1, .1, 1],
                        help='loss coefficients for level 6, 7, and 6+7 (default: [.1, .1, 1])')

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
    parser.add_argument('-detach', '--detach', action='store_true',
                        help='Detach in loss?')
    parser.add_argument('-monoreg', '--monotonicity_regularization', action='store_true',
                        help='use monotonicity_regularization?')
    parser.add_argument('-g', '--gamma', type=float, default=0.1,
                        help='coefficient of monotonicity regularization (default: 0.1)')

    parser.add_argument('-ds', '--dataset', default='KonIQ-10k', type=str,
                        help='dataset name (default: KonIQ-10k)')
    parser.add_argument('-eid', '--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('-tr', '--train_ratio', type=float, default=0.6,
                        help='train ratio (default: 0.6)')
    parser.add_argument('-tvr', '--train_and_val_ratio', type=float, default=0.8,
                        help='train_and_val_ratio (default: 0.8)')

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

    parser.add_argument('-logd', "--log_dir", type=str, default="runs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('-tdt', '--test_during_training', action='store_true',
                        help='test_during_training?')  # It is better to re-make a train_loader_for_evaluation so as not to disturb the random number generator.
    parser.add_argument('-eval', '--evaluate', action='store_true',
                        help='Evaluate only?')

    parser.add_argument('-debug', '--debug', action='store_true',
                        help='Debug the training by reducing dataflow to 5 batches')
    parser.add_argument('-pbar', '--pbar', action='store_true',
                        help='Use progressbar for the training')

    args = parser.parse_args()
    if args.lr_decay == 1 or args.epochs < 3:  # no lr decay
        args.lr_decay_step = args.epochs
    else:  # 
        args.lr_decay_step = int(args.epochs/(1+np.log(args.overall_lr_decay)/np.log(args.lr_decay)))

    # KonIQ-10k that train-val-test split provided by the owner
    if args.dataset == 'KonIQ-10k':
        args.train_ratio = 7058/10073
        args.train_and_val_ratio = 8058/10073
        if not args.resize:
            args.resize_size_h = 768
            args.resize_size_w = 1024

    if args.beta[1] + args.beta[-1] == .0:
        args.val_criterion = 'SROCC1'
    if args.beta[0] + args.beta[-1] == .0:
        args.val_criterion = 'SROCC2'

    args.im_dirs = {'KonIQ-10k': 'KonIQ-10k',  
                    'CLIVE': 'CLIVE' 
                    }  # ln -s database_path xxx
    args.data_info = {'KonIQ-10k': './data/KonIQ-10kinfo.mat',
                      'CLIVE': './data/CLIVEinfo.mat'}

    if not args.randomness:
        torch.manual_seed(args.seed)  #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    args.format_str = '{}-{}-bn_end={}-loss={}-p={}-q={}-detach-{}-ft_lr_ratio={}-alpha={}-beta={}-{}-res={}-{}x{}-aug={}-monotonicity={}-lr={}-bs={}-e={}-opt_level={}-EXP{}'\
                      .format(args.architecture, args.pool, args.use_bn_end, args.loss_type, args.p, args.q, args.detach, args.ft_lr_ratio, args.alpha, args.beta, 
                              args.dataset, args.resize, args.resize_size_h, args.resize_size_w, args.augmentation, 
                              args.monotonicity_regularization, args.learning_rate, args.batch_size, args.epochs, args.opt_level, args.exp_id)
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    args.trained_model_file = 'checkpoints/' + args.format_str
    if not os.path.exists('results'):
        os.makedirs('results')
    args.save_result_file = 'results/' + args.format_str
    print(args)
    run(args)
