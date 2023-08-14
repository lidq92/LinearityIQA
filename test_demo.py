# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2020/1/14

import torch
from IQAmodel import IQAModel
import os
import numpy as np
import random
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import h5py

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQAModel(arch=args.architecture, pool=args.pool, use_bn_end=args.use_bn_end, P6=args.P6, P7=args.P7).to(device)  #
    im = Image.open(args.img_path).convert('RGB')  #
    if args.resize:  # resize or not?
        im = resize(im, (args.resize_size_h, args.resize_size_w)) #
    im = to_tensor(im).to(device)
    im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

    checkpoint = torch.load(args.trained_model_file)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        y = model(im.unsqueeze(0))
    k = checkpoint['k']
    b = checkpoint['b']
    print('The image quality score is {}'.format(y[-1].item() * k[0] + b[0])) # See line 39 of IQAperformance.py, i=0

if __name__ == "__main__":
    parser = ArgumentParser(description='Test Demo for LinearityIQA')

    parser.add_argument('--architecture', default='resnext101_32x8d', type=str,
                        help='arch name (default: resnext101_32x8d)')
    parser.add_argument('--pool', default='avg', type=str,
                        help='pool method (default: avg)')
    parser.add_argument('--use_bn_end', action='store_true',
                        help='Use bn at the end of the output?')
    parser.add_argument('--P6', type=int, default=1,
                        help='P6 (default: 1)')
    parser.add_argument('--P7', type=int, default=1,
                        help='P7 (default: 1)')

    parser.add_argument('--trained_model_file', default='checkpoints/p1q2.pth', type=str,
                        help='trained_model_file')

    parser.add_argument('--img_path', default='data/1000.JPG', type=str,
                        help='test image path')
    parser.add_argument('--resize', action='store_true',
                        help='Resize?')
    parser.add_argument('--resize_size_h', default=498, type=int,
                        help='resize_h (default: 498)')
    parser.add_argument('--resize_size_w', default=664, type=int,
                        help='resize_w (default: 664)')

    args = parser.parse_args()

    run(args)
