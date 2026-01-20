import os

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataReader(Dataset):
    def __init__(self, img_dir, inp='input', tar='target', mode='train', ori=False, img_options=None):
        super(DataReader, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(img_dir, inp)))
        mask_files = sorted(os.listdir(os.path.join(img_dir, 'mask')))
        tar_files = sorted(os.listdir(os.path.join(img_dir, tar)))

        self.inp_filenames = [os.path.join(img_dir, inp, x) for x in inp_files if is_image_file(x)]
        self.mask_filenames = [os.path.join(img_dir, 'mask', x) for x in mask_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(img_dir, tar, x) for x in tar_files if is_image_file(x)]

        self.mode = mode

        self.img_options = img_options

        self.sizex = len(self.tar_filenames)  # get the size of target

        if self.mode == 'train':
            self.transform = A.Compose([
                A.Flip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(p=0.3),
                A.Transpose(p=0.3),
                A.RandomResizedCrop(height=img_options['h'], width=img_options['w']),
            ],
                additional_targets={
                    'target': 'image',
                    'mask': 'image',
                }
            )
            # self.degrade = A.Compose([
            #     # A.RandomShadow(p=0.5)
            #     A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=10, shadow_dimension=15, p=1)
            # ])
        else:
            if ori:
                self.transform = A.Compose([
                    A.NoOp(),
                ],
                    additional_targets={
                        'target': 'image',
                        'mask': 'image',
                    }
                )
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_options['h'], width=img_options['w']),
                ],
                    additional_targets={
                        'target': 'image',
                        'mask': 'image',
                    }
                )
            self.degrade = A.Compose([
                A.NoOp(),
            ])

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        tar_path, transformed = self.load(index_)
        
        inp_img = F.to_tensor(transformed['image'])
        mas_img = F.to_tensor(transformed['mask'])
        tar_img = F.to_tensor(transformed['target'])

        filename = os.path.basename(tar_path)

        return inp_img, tar_img, filename, mas_img

    def load(self, index_):
        inp_path = self.inp_filenames[index_]
        mask_path = self.mask_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = np.array(inp_img)
        mask_img = np.array(mask_img)
        tar_img = np.array(tar_img)

        transformed = self.transform(image=inp_img, target=tar_img, mask=mask_img)

        return tar_path, transformed