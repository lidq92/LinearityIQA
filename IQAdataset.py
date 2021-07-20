from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize, rotate, crop, hflip, to_tensor, normalize
from PIL import Image
import h5py
import os
import numpy as np
import random


def default_loader(path):
    return Image.open(path).convert('RGB')  #


class IQADataset(Dataset):
    def __init__(self, args, status='train', loader=default_loader):
        self.status = status

        self.augment = args.augmentation
        self.angle = args.angle
        self.crop_size_h = args.crop_size_h
        self.crop_size_w = args.crop_size_w
        self.hflip_p = args.hflip_p

        Info = h5py.File(args.data_info[args.dataset], 'r')
        index = Info['index']
        index = index[:, args.exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]
        if status == 'train':
            index = index[0:int(args.train_ratio * len(index))]
        elif status == 'val':
            index = index[int(args.train_ratio * len(index)):int(args.train_and_val_ratio * len(index))]
        elif status == 'test':
                index = index[int(args.train_and_val_ratio * len(index)):len(index)]
        self.index = []
        for i in range(len(ref_ids)):
            if ref_ids[i] in index:
                self.index.append(i)
        print("# {} images: {}".format(status, len(self.index)))

        self.label = Info['subjective_scores'][0, self.index].astype(np.float32)
        self.label_std = Info['subjective_scoresSTD'][0, self.index].astype(np.float32)
        self.im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]
        self.ims = []
        for im_name in self.im_names:
            im = loader(os.path.join(args.im_dirs[args.dataset], im_name))
            if args.dataset == 'CLIVE':  #
                w, h = im.size
                if w != 500 or h != 500:
                    im = resize(im, (500, 500))  #
            if args.resize:  # resize or not?
                im = resize(im, (args.resize_size_h, args.resize_size_w))  # h, w
            self.ims.append(im)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.transform(self.ims[idx], self.status, self.angle, self.crop_size_h, self.crop_size_w, self.hflip_p)
        label = self.label[idx]
        label_std = self.label_std[idx]
        return im, (label, label_std)

    def transform(self, im, status, angle=2, crop_size_h=498, crop_size_w=498, hflip_p=0.5):
        if status == 'train' and self.augment:  # data augmentation
            angle = random.uniform(-angle, angle)
            p = random.random()
            w, h = im.size
            i = random.randint(0, h - crop_size_h)
            j = random.randint(0, w - crop_size_w)

            im = rotate(im, angle)
            if p < hflip_p:
                im = hflip(im)
            im = crop(im, i, j, self.crop_size_h, self.crop_size_w)
        im = to_tensor(im)
        im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        return im


def get_data_loaders(args):
    """ Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, val_loader, test_loader
    """
    train_dataset = IQADataset(args, 'train')
    batch_size = args.batch_size
    if args.debug:
        num_samples = 5 * batch_size
        print("Debug mode: reduced training dataset to the first {} samples".format(num_samples))
        train_dataset = Subset(train_dataset, list(range(num_samples)))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)  # If the last batch only contains 1 sample, you need drop_last=True.
    val_dataset = IQADataset(args, 'val')
    test_dataset = IQADataset(args, 'test')
    if args.debug:
        num_samples = 5
        print("Debug mode: reduced validation/test datasets to the first {} samples".format(num_samples))
        val_dataset = Subset(val_dataset, list(range(num_samples)))
        test_dataset = Subset(test_dataset, list(range(num_samples)))

    val_loader = DataLoader(val_dataset)    
    test_loader = DataLoader(test_dataset)
    return train_loader, val_loader, test_loader
