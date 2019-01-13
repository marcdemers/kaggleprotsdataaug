

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scipy.optimize as opt

import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from collections import Counter
import torch
from PIL import Image
from skimage import io
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import PIL
import time
path_local = 'all/'
path_drive = '/media/marc/"Expansion Drive"/KaggleProtsData/train_full_size/'
path_drive_test = '/media/marc/"Expansion Drive"/KaggleProtsData/test_full_size/'
extension = '.png'#''.tif' #'.png'

import threading

class TrainProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, oversampling=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv(path_local + 'train.csv')
        percent = 1 # 0.8  # default .95
        train = Counter()
        for i, ii in self.prots_df.iterrows():
            train.update(ii['Target'].split())

        # good_val = False
        # idx = 0
        # while good_val is False:
        #     idx += 1
        #     good_val = True
        #     self.prots_df = self.prots_df.sample(frac=1)
        #     self.validation_df = self.prots_df[int(percent * len(self.prots_df)):]
        #     val = Counter()
        #     for i, ii in self.validation_df.iterrows():
        #
        #         val.update(ii['Target'].split())
        #
        #     for i in range(28):
        #         if val[str(i)] == 0:
        #             good_val = False
        #         val_ratio = val[str(i)] / len(self.validation_df)
        #         train_ratio = train[str(i)] / int(percent * len(self.prots_df))
        #
        #         if ((1 / 1.75) < val_ratio / train_ratio < 1.75) is False:
        #             good_val = False
        #         # print(val_ratio / train_ratio, (1 / 1.75) < val_ratio / train_ratio < 1.75, good_val)
        #
        # self.validation_df.to_csv('validation.csv', index=False)
        self.prots_df = self.prots_df[:int(percent * len(self.prots_df))]

        self.oversampling(oversampling)

        self.targets = torch.zeros(size=(len(self.prots_df), 28), dtype=torch.float)
        for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: list(map(int, x.split()))).values.tolist()):
            for cat in cats:
                self.targets[i, cat] = 1

        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

        # print(self.prots_df['Id'])
        # print(self.prots_df['Target'])
        # print("----")
        # print(pd.DataFrame.sample(self.prots_df))
        # print(pd.DataFrame.sample(self.prots_df).values.tolist())
    def data_augHconcat(self, img1, img1_labels, normalize=False):#, img2, img2_labels):
        img2, img2_labels = self.sample_second_image(normalize)
        if normalize == True:
            img1 = np.asarray(img1 - np.mean(img1, axis=(1, 2), keepdims=True), dtype=np.uint8)
        x = 1#np.random.beta(1,1)
        r = 1 - x
        img_mixed = np.concatenate((img1[:, :, :int(r * img1.shape[2])], img2[:, :, int(r * img2.shape[2]):]), axis=2)

        mixed_img_labels = img1_labels * r
        for i, item in enumerate(img2_labels):  # img2_labels is a list
            if mixed_img_labels[item] == 0:
                mixed_img_labels[item] = 1 - r
            else:
                if (1 - r) + mixed_img_labels[item] <= 1:
                    mixed_img_labels[item] = (1 - r) + mixed_img_labels[item]
                else:
                    mixed_img_labels[item]

        return img_mixed, mixed_img_labels

    def data_augVconcat(self, img1, img1_labels,normalize=False):#, img2, img2_labels):
        # print("H concat")
        # img1, img2 :already numpy array
        img2, img2_labels = self.sample_second_image(normalize)
        if normalize == True:
            img1 = np.asarray(img1 - np.mean(img1, axis=(1, 2), keepdims=True), dtype=np.uint8)
        x = 1#np.random.beta(1, 1)
        r=1-x
        img_mixed = np.concatenate((img1[:, :int(r * img1.shape[1]), :], img2[:, int(r * img2.shape[1]):, :]), axis=1)

        mixed_img_labels = img1_labels * r

        for i, item in enumerate(img2_labels):  # img2_labels is a list
            if mixed_img_labels[item] == 0:
                mixed_img_labels[item] = 1 - r
            else:
                if (1 - r) + mixed_img_labels[item] <= 1:
                    mixed_img_labels[item] = (1 - r) + mixed_img_labels[item]
                else:
                    mixed_img_labels[item] = 1

        return img_mixed, mixed_img_labels


    def data_Mixup_two_images(self, img1, img1_labels, img2, img2_labels):  # mixes image2 into image1
        # with probability r
        r = np.random.beta(0.9, 0.9)
        img_mixed = np.asarray(r * img1 + (1 - r) * img2, dtype=np.uint8)

        #labels
        mixed_img_labels = img1_labels * r + (1-r) * img2_labels

        return img_mixed, mixed_img_labels

    def data_augBC(self, img1, img1_labels, img2, img2_labels):  # mixes image2 into image1
        #input images must be mean-substracted before
        # mix the images with BC+
        # with probability p
        r = np.random.beta(1, 1)#uniform for BC+

        sigma1 = np.std(img1, axis=(1, 2), keepdims=True)
        sigma2 = np.std(img2, axis=(1, 2), keepdims=True)
        p = 1 / (1 + sigma1 * (1 - r) / (sigma2 * r))
        img_mixed = np.asarray( (p * img1 + (1 - p) * img2) / (np.sqrt(p ** 2 + (1 - p) ** 2)) , dtype=np.uint8)

        # labels
        label_mixing_coef = np.mean(p)
        mixed_img_labels = img1_labels * label_mixing_coef + (1 - label_mixing_coef) * img2_labels

        return img_mixed, mixed_img_labels

    def oversampling(self, ratio=100):
        c = Counter()
        for i, ii in self.prots_df.iterrows():
            c.update(ii['Target'].split())

        self.ratios = np.array([c[str(i)] / len(self.prots_df) for i in range(28)])

        max_v = 0
        for i in c:
            if c[i] > max_v:
                max_v = c[i]
        # print(max_v, ratio)
        max_v = max_v / ratio
        extra_x, extra_y = [], []
        for value in range(28):

            oversample = int(max_v * (1 - (c[str(value)] / max_v)))
            relevant_indexes = []
            for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: x.split()).values.tolist()):
                if str(value) in cats:
                    relevant_indexes.append(i)
            indexes = [random.choice(relevant_indexes) for _ in range(oversample)]

            for idx in indexes:
                extra_x.append(self.prots_df['Id'].values[idx])
                extra_y.append(self.prots_df['Target'].values[idx])

        x = np.concatenate((self.prots_df['Id'].values, extra_x), axis=0)
        y = np.concatenate((self.prots_df['Target'], extra_y), axis=0)
        self.prots_df = pd.DataFrame(columns=['Id', 'Target'])
        self.prots_df['Id'] = x
        self.prots_df['Target'] = y

    def sample_second_image(self, normalize=False):
        samples = np.array(pd.DataFrame.sample(self.prots_df,4).values.tolist())
        img2_path = (samples[:, 0]).tolist()
        img2_name = [path_local + 'train/' + img for img in img2_path]
        img2 = np.array([np.array([io.imread(img2_name[i] + '_' + color + extension) for color in self.color], dtype=np.uint8) for i in range(4)])

        img2_labels = (samples[:,1]).tolist()
        img2_labels_list = [list(map(int, elem.split(" "))) for elem in img2_labels]

        if normalize == True:
            img2[0] = np.asarray(img2[0] - np.mean(img2[0], axis=(1, 2), keepdims=True), dtype=np.uint8)

        return img2[0], img2_labels_list[0]

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = path_local+'train/' + self.prots_df.iloc[idx, 0]
        img_labels = self.targets[idx]
        #sample from uniform distro
        # Y= np.random.beta(1,1)
        # # print(Y)
        # # if Y <= 0.5:
        img = np.array([io.imread(img_name + '_' + color + extension) for color in self.color], dtype=np.uint8)
        #
        # # data augmentation - comment if undesireable
        img_VC, img_VC_labels = self.data_augVconcat(img, img_labels, normalize=False)# , img2, img2_labels_list)
        img_HC, img_HC_labels = self.data_augHconcat(img, img_labels, normalize=False)  # , img2, img2_labels_list)

        img_mixed, img_mixed_labels = self.data_Mixup_two_images(img_VC, img_VC_labels, img_HC, img_HC_labels)
        # self.show_images(img_mixed)

        #preparing the return
        img = self.transform(Image.fromarray(np.swapaxes(img_mixed, 0, 2)))
        img_labels = img_mixed_labels

        # else:
        #     img = self.transform(Image.fromarray(
        #     np.swapaxes(np.array([io.imread(img_name + '_' + color + '.png') for color in self.color], dtype=np.uint8),
        #                0, 2)))

        return img, img_labels

    def show_images(self, img):
        red = Image.fromarray(img[0, :, :])
        green = Image.fromarray(img[1, :, :])
        blue = Image.fromarray(img[2, :, :])
        yellow = Image.fromarray(img[3, :, :])
        rgb = PIL.Image.merge('RGB', (red, green, blue))
        y = PIL.Image.merge('RGB', (yellow, yellow, PIL.Image.new('L', (yellow.width, yellow.height))))
        rgby = PIL.ImageChops.add(rgb, y)
        plt.imshow(np.asarray(rgby))
        plt.axis('off')
        plt.show()






class ValProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv('validation.csv')
        self.targets = torch.zeros(size=(len(self.prots_df), 28), dtype=torch.float)
        for i, cats in enumerate(self.prots_df['Target'].apply(lambda x: list(map(int, x.split()))).values.tolist()):
            for cat in cats:
                self.targets[i, cat] = 1

        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = path_local + 'train/'+ self.prots_df.iloc[idx, 0]
        img_labels = self.targets[idx]

        # img = self.transform(Image.fromarray(
        #    (np.array(io.imread(img_name + '_' + 'green' + extension), dtype=np.uint8))))

        # img_temp = np.array([io.imread(img_name + '_' + color + extension) for color in self.color], dtype=np.uint8)
        # img = np.asarray( (img_temp - np.mean(img_temp, axis=(1, 2), keepdims=True)/np.std(img_temp, axis=(1, 2), keepdims=True)), dtype=np.uint8)
        # # self.show_images(img)
        # img = np.swapaxes(img,0,2)
        #
        #
        # img = self.transform(Image.fromarray(img))

        img = self.transform(Image.fromarray(
            np.swapaxes(
                np.array([io.imread(img_name + '_' + color + extension) for color in self.color], dtype=np.uint8),
                0, 2)))


        return img, img_labels

    def show_images(self, img):
        red = Image.fromarray(img[0, :, :])
        green = Image.fromarray(img[1, :, :])
        blue = Image.fromarray(img[2, :, :])
        yellow = Image.fromarray(img[3, :, :])
        rgb = PIL.Image.merge('RGB', (red, green, blue))
        y = PIL.Image.merge('RGB', (yellow, yellow, PIL.Image.new('L', (yellow.width, yellow.height))))
        rgby = PIL.ImageChops.add(rgb, y)
        plt.imshow(np.asarray(rgby))
        plt.axis('off')
        plt.show()


class TestProtsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.prots_df = pd.read_csv('all/sample_submission.csv')

        self.transform = transform
        self.color = ['red', 'green', 'blue', 'yellow']

    def __len__(self):
        return len(self.prots_df)

    def __getitem__(self, idx):
        img_name = 'all/test/' + self.prots_df.iloc[idx, 0]
        img = self.transform(Image.fromarray(
            np.swapaxes(np.array([io.imread(img_name + '_' + color + extension) for color in self.color], dtype=np.uint8),
                        0, 2)))
        return img


