## Results in the Paper
Note: To avoid Type 3 fonts in the matplotlib plots, one can change the default `pdf.fonttype : 3` to `pdf.fonttype : 42`. After producing the plots, they may have large file sizes. One can then compress the files on [ilovepdf](https://www.ilovepdf.com/compress_pdf).
### Figure 1
```bash 
python main.py --resize --loss_type mae --test_during_training 
python main.py --resize --loss_type mse --test_during_training
python main.py --resize --loss_type norm-in-norm --test_during_training
```
After completing the above commands, you can download the performance metric (SROCC/PLCC/RMSE) values in each stage (train/val/test) and for each loss (mae/mse/norm-in-norm) from TensorBoard visualization (saved to `'results_in_the_paper/csv/loss={}-{}_KonIQ-10k_{}.csv'.format(loss, stage, metric)`).
```bash
cd results_in_the_paper
python loss_performance_curves.py # Plot Figure 1
cd ..
```

### Figure 4
```bash 
# python main.py --resize --loss_type norm-in-norm --test_during_training # done before
# python main.py --resize --loss_type mse --test_during_training # done before
python main.py --resize --loss_type mse --test_during_training --use_bn_end 
```
After completing the above commands, you can download the performance metric (SROCC/PLCC) values in the val stage and for each loss (MSE/bnMSE/Norm-in-Norm) from TensorBoard visualization (saved to `'results_in_the_paper/csv/{}_{}.csv'.format(loss, metric)`).
```bash
cd results_in_the_paper
python bnMSE.py # Plot Figure 4
cd ..
```
### Figure 5
```bash 
## base exp
python main.py --resize --lr 1e-4 -bs 8 --ft_lr_ratio 0.1 --arch resnet50
# other lr
python main.py --resize --lr 1e-3 --arch resnet50
python main.py --resize --lr 1e-5 --arch resnet50
# other bs
python main.py --resize -bs 4 --arch resnet50
python main.py --resize -bs 16 --arch resnet50
# other ft_lr_ratio
python main.py --resize --ft_lr_ratio 0 --arch resnet50
python main.py --resize --ft_lr_ratio 0.01 --arch resnet50
python main.py --resize --ft_lr_ratio 1 --arch resnet50
# image_size (not shown in the paper)
###
```
After completing the above commands, you can download the PLCC values in the val stage for the norm-in-norm loss and for each parameter (lr/bs/ft_lr_ratio) value from TensorBoard visualization (saved to `'results_in_the_paper/csv/loss=norm-in-norm-{}={}-val_KonIQ-10k_PLCC.csv'.format(parameter, value)`).
```bash
# Plot Figure 5
cd results_in_the_paper
python lr_performance_plot.py
python batch_size_performance_plot.py
python ft_lr_ratio_performance_plot.py
# python image_size_performance_plot.py
cd ..
```

### Figure 6 & 7
```bash
## Training on KonIQ-10k train set
python main.py --resize --arch resnet18
python main.py --resize --arch resnet34
# python main.py --resize --arch resnet50 # done before
# python main.py --resize --arch resnext101_32x8d # done before
## Testing on KonIQ-10k test set and CLIVE using test_dataset.py
python test_dataset.py --resize --arch resnet18
python test_dataset.py --resize --arch resnet34
python test_dataset.py --resize --arch resnet50 
python test_dataset.py --resize --arch resnext101_32x8d 
python test_dataset.py --resize --arch resnet18 --dataset CLIVE
python test_dataset.py --resize --arch resnet34 --dataset CLIVE
python test_dataset.py --resize --arch resnet50 --dataset CLIVE
python test_dataset.py --resize --arch resnext101_32x8d --dataset CLIVE 
```

After completing the above commands, you can download the PLCC values in the val stage for the norm-in-norm loss and for each parameter (lr/bs/ft_lr_ratio) value from TensorBoard visualization (saved to `'results_in_the_paper/csv/loss=norm-in-norm-{}={}-val_KonIQ-10k_PLCC.csv'.format(parameter, value)`).
```bash
cd results_in_the_paper
python backbone_performance_plot.py # plot Figure 6
python scatter_plots.py # plot Figure 7
cd ..
```

### Table 1
```bash
python main.py --resize --arch resnet18 --p 1 --q 1
python main.py --resize --arch resnet18 --p 1 --q 2
python main.py --resize --arch resnet18 --p 2 --q 1
python main.py --resize --arch resnet18 --p 2 --q 2
python main.py --resize --arch resnet34 --p 1 --q 1
python main.py --resize --arch resnet34 --p 1 --q 2
python main.py --resize --arch resnet34 --p 2 --q 1
python main.py --resize --arch resnet34 --p 2 --q 2
python main.py --resize --arch resnet50 --p 1 --q 1
python main.py --resize --arch resnet50 --p 1 --q 2
python main.py --resize --arch resnet50 --p 2 --q 1
python main.py --resize --arch resnet50 --p 2 --q 2
python main.py --resize --arch resnext101_32x8d --p 1 --q 1
python main.py --resize --arch resnext101_32x8d --p 1 --q 2
python main.py --resize --arch resnext101_32x8d --p 2 --q 1
python main.py --resize --arch resnext101_32x8d --p 2 --q 2
```

### Table 2
```bash
python main.py --resize # done before
python main.py --resize --alpha 1 0.1 
python test_dataset.py --dataset KonIQ-10k --resize
python test_dataset.py --dataset KonIQ-10k --resize --alpha 1 0.1
python test_dataset.py --dataset CLIVE --resize
python test_dataset.py --dataset CLIVE --resize --alpha 1 0.1
# Testing on KonIQ-10k test set and CLIVE using test_dataset.py by specifying the settings and the trained_model_file.
## If you have downloaded the pre-trained model weights, you can also run the test with the following commands
# python test_dataset.py --dataset KonIQ-10k --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2.pth
# python test_dataset.py --dataset KonIQ-10k --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2plus0.1variant.pth
# python test_dataset.py --dataset CLIVE --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2.pth
# python test_dataset.py --dataset CLIVE --resize --arch resnext101_32x8d --trained_model_file checkpoints/p1q2plus0.1variant.pth
```

Note: Due to a wrong implementation of `rho`'s calculation, the result of `𝑙 + 0.1𝑙′` is no longer the reported one. It should be `SROCC=0.937, PLCC=0.946` on KonIQ-10k and `SROCC=0.834, PLCC=0.850` on CLIVE. Additionally, the result of `𝑙′` is `SROCC=0.937, PLCC=0.947` on KonIQ-10k and `SROCC=0.831, PLCC=0.847` on CLIVE. And the result of `𝑙+𝑙′` is `SROCC=0.937, PLCC=0.947` on KonIQ-10k and `SROCC=0.836, PLCC=0.848` on CLIVE. 
### Figure A1 & A2
Uncomment line 96 of `IQAloss.py` to print bhat (i.e., $\hat{b}$).
```bash
python main.py --resize > bhat.log 2>&1 & 
```
```bash
cd results_in_the_paper
python bhat.py
cd ..
```

### Table A1
In line 37-43 of `main.py`, `Adam` can be changed to other optimizers, such as `SGD`, `Adadelta`. Other settings are same as the settings for Figure 5(a).

### Table A2
Set `--arch` to `alexnet` or `vgg16`. Other settings are same as the settings for Figure 6.

### t-test
Set `--arch` to `resnet18` and `--dataset` to `CLIVE`. Run experiments from `--exp_id=0` to `--exp_id=9`.
Then conduct paired t-test based on PLCC values over ten runs.