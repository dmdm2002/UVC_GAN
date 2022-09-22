import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random


class Loader(data.DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        super(Loader, self).__init__(self)
        self.dataset_dir = dataset_dir
        self.styles = styles
        folder_A = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')

        self.transform = transforms

        self.image_path_A = []
        self.image_path_B = []

        for i in range(len(folder_A)):
            A = glob.glob(f'{folder_A[i]}/*.png')
            B = glob.glob(f'{folder_B[i]}/*.png')
            B = self.shuffle_folder(A, B)

            self.image_path_A = self.image_path_A + A
            self.image_path_B = self.image_path_B + B

    def shuffle_folder(self, A, B):
        random.shuffle(B)
        for i in range(len(A)):
            if A[i] == B[i]:
                return self.shuffle_folder(A, B)
        return B

    def __getitem__(self, index_A):
        index_B = random.randint(0, len(self.image_path_B)-1)

        item_A = self.transform(Image.open(self.image_path_A[index_A]))
        item_B = self.transform(Image.open(self.image_path_B[index_B]))

        return [item_A, item_B, self.image_path_A[index_A]]

    def __len__(self):
        return len(self.image_path_A)