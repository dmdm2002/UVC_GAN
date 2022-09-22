import Options
import os
import re
import glob
import torch
import torchvision.transforms as transforms
import random
import PIL.Image as Image
import numpy
import math
from Model.Unet_Generator import UNet_vit
import torch
import torch.nn as nn
import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance as FID


class test(Options.param):
    def __init__(self):
        super(test, self).__init__()

    def psnr(self, img1, img2):
        mse = numpy.mean((img1 - img2) ** 2)  # MSE 구하는 코드
        # print("mse : ", mse)
        if mse == 0:
            return 100

        PIXEL_MAX = 255.0

        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))  # PSNR구하는 코드

    def np2img(self, output, transform_to_image, name, epoch, type):
        os.makedirs(f"{self.OUTPUT_SAMPLE}/{epoch}/A/{type}", exist_ok=True)
        output = transform_to_image(output.squeeze())
        output.save(f'{self.OUTPUT_SAMPLE}/{epoch}/A/{type}/{name}.png')

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'[device] : {device}')
        print('--------------------------------------------------------------------------------')

        # 1. Model Build
        netG_A2B = UNet_vit(3).to(device)
        netG_B2A = UNet_vit(3).to(device)

        # 2. Load CKP
        # checkpoint = torch.load(f'{self.OUTPUT_CKP}/ckp/199.pth', map_location=device)
        # netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
        # netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])
        #
        # netG_A2B.eval()
        # netG_B2A.eval()

        # 3. Load DataSets
        transform_to_tensor = transforms.Compose(
            [
                transforms.Resize((self.SIZE, self.SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        transforms_to_image = transforms.Compose(
            [
                transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                transforms.ToPILImage(),
            ]
        )

        # 4. Setting Folder
        os.makedirs(f'{self.OUTPUT_TEST}/A2B', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A', exist_ok=True)

        os.makedirs(f'{self.OUTPUT_TEST}/A2B_LOSS', exist_ok=True)
        os.makedirs(f'{self.OUTPUT_TEST}/B2A_LOSS', exist_ok=True)

        test_list = [["A2B", netG_A2B]]

        # LOOP

        for epoch in range(113, 114):
            full_psnr = 0
            fid = FID(feature=2048).to(device)
            checkpoint = torch.load(f'{self.OUTPUT_CKP}/ckp/{epoch}.pth', map_location=device)
            netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
            netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])

            netG_A2B.eval()
            netG_B2A.eval()

            for folder_name, model in test_list:
                print(f'[Folder Name] : {folder_name}')

                A_folders = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0])}/*/*')
                B_folders = glob.glob(f'{os.path.join(self.DATASET_PATH, self.DATA_STYPE[0])}/*/*')
                # print(A_folders)

                if folder_name == "A2B":
                    image_path_list = A_folders
                else:
                    image_path_list = B_folders

                for idx, img_path in enumerate(tqdm.tqdm(image_path_list)):
                    # print(f'|| {idx}/{len(image_path_list)} ||')
                    image = Image.open(img_path)
                    # print(type(image))
                    image = transform_to_tensor(image).unsqueeze(0)
                    image = image.to(device)

                    output = model(image)

                    # print(image.type(torch.uint8).dtype)

                    fid.update(image.type(torch.uint8), real=True)
                    fid.update(output.type(torch.uint8), real=False)

                    output = output.cpu().squeeze().detach().numpy()
                    image = image.cpu().squeeze().numpy()

                    psnr = self.psnr(image, output)
                    full_psnr = full_psnr + psnr

                full_psnr = full_psnr / len(image_path_list)
                fid_score = fid.compute()
                print(f'===> [{epoch}/200]  PSNR : {full_psnr}  FID : {fid_score}')

