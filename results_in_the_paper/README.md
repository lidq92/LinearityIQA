## Results in the Paper
### Figure 1
```bash 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type l1 --test_during_training # Add --test_during_training if you want to show train/test performance values in TensorBoard during the training.
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type mse --test_during_training
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 --test_during_training
```
After running the above commands, you can download the performance values in each epoch from TensorBoard visualization (saved to `'csv/loss={}-{}_KonIQ-10k_{}.csv'.format(loss, stage, metric)`).
```bash
python loss_performance_curves.py # Plot Figure 1
```

### Figure 4
```bash 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type mse --test_during_training 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type mse --test_during_training --use_bn_end 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 --test_during_training 
```
After running the above commands, you can download the performance values in each epoch from TensorBoard visualization (saved to `'csv/{}_{}.csv'.format(loss, metric)`).
```bash
python bnMSE.py # Plot Figure 4
```
### Figure 5
```bash 
# lr
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-3 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2# Note: basic experiment 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-5 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2
# bs
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2# Basic. No need to run again.
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2
# ft_lr_ratio
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 0 --arch resnet50 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.01 --arch resnet50 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2# Basic. No need to run again.
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 4 -e 30 --ft_lr_ratio 1 --arch resnet50 --loss_type Lp --p 1 --q 2
# image_size (not shown in the paper)
### 
```
After running the above commands, you can download the performance values in each epoch from TensorBoard visualization.
```bash
 # Plot Figure 5
python lr_performance_plot.py
python batch_size_performance_plot.py
python ft_lr_ratio_performance_plot.py
# python image_size_performance_plot.py
```

### Figure 6 & 7
```bash
# Training
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnet18 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnet34 --loss_type Lp --p 1 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 16 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 
# Testing on KonIQ-10k test set and CLIVE using test_dataset.py by specifying the settings and the trained_model_file.
# ...
```
```bash
 # Plot Figure 6
python backbone_performance_plot.py
# Plot Figure 7
python scatter_plots.py # (predicted quality, mos) are saved in `../results/`, and we copy and rename them to `npy/`
```

### Table 1
```bash
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 2 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 2 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 2 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet50 --loss_type Lp --p 2 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet34 --loss_type Lp --p 1 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet34 --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet34 --loss_type Lp --p 2 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet34 --loss_type Lp --p 2 --q 2
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet18 --loss_type Lp --p 1 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet18 --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet18 --loss_type Lp --p 2 --q 1
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnet18 --loss_type Lp --p 2 --q 2
```

### Table 2
```bash
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 
python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 --alpha 1 0.1 
# Testing on KonIQ-10k test set and CLIVE using test_dataset.py by specifying the settings and the trained_model_file.
# ... If you have renamed the model files, then refer to the Testing section in root README.md
CUDA_VISIBLE_DEVICES=0 python test_dataset.py --dataset KonIQ-10k --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2.pth
CUDA_VISIBLE_DEVICES=1 python test_dataset.py --dataset KonIQ-10k --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2plus0.1variant.pth
CUDA_VISIBLE_DEVICES=0 python test_dataset.py --dataset CLIVE --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2.pth
CUDA_VISIBLE_DEVICES=1 python test_dataset.py --dataset CLIVE --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2plus0.1variant.pth
```

### Figure A1 & A2
Uncomment line 96 of `IQAloss.py` to print bhat (i.e., $\hat{b}$).
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset KonIQ-10k --resize --exp_id 0 --lr 1e-4 -bs 8 -e 30 --ft_lr_ratio 0.1 --arch resnext101_32x8d --loss_type Lp --p 1 --q 2 > bhat.log 2>&1 & 
```
```bash
python bhat.py
```

### Table A1
In line 37-43 of `main.py`, `Adam` can be changed to other optimizers, such as `SGD`, `Adadelta`. Other settings are same as  the settings for Figure 5(a).

### Table A2
Set `--arch` to `alexnet` or `vgg16`. Other settings are same as the settings for Figure 6.

### t-test
Set `--arch` to `resnet18` and `--dataset` to `CLIVE`. Run experiments from `--exp_id=0` to `--exp_id=10`.
Then conduct paired t-test based on PLCC values over ten runs.