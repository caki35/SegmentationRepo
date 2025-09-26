import torch
from torch.utils.data import Dataset
import os
import re
import numpy as np
import cv2
import random
from skimage.color import rgb2hed
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import staintools
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
np.bool = bool
import imgaug.augmenters as iaa
import torchio as tio

image_ext = ['.jpg', '.jpeg', '.webp', '.bmp', '.png', '.tif', '.PNG', '.tiff']



def RadiologyAugmentationTIO(sample, transforms_dict):
    image, label = sample['image'], sample['label']
    
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=np.expand_dims(image,(0,-1))),  # Add channel and batch dim
        label=tio.LabelMap(tensor=np.expand_dims(label,(0,-1))) # Add channel and batch dim
    )  
    # Apply augmentations
    transform = tio.OneOf(transforms_dict)
    transformed_subject = transform(subject)
    
    transformed_image = transformed_subject["image"].data.numpy()[0,:,:,0]
    transformed_label = transformed_subject["label"].data.numpy()[0,:,:,0]
    sample = {'image': transformed_image, 'label': transformed_label}
    return sample



def random_rot_flip(sample_list):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    res_list = []
    for currentSample in sample_list:
        currentSample = np.rot90(currentSample, k)
        currentSample = np.flip(currentSample, axis=axis).copy()
        res_list.append(currentSample)
    return res_list


def random_rotate(sample_list):
    angle = np.random.randint(-20, 20)
    res_list = []
    for currentSample in sample_list:
        currentSample = ndimage.rotate(currentSample, angle, order=0, reshape=False)
        res_list.append(currentSample)
    return res_list


class DataSingle(Dataset):
    def __init__(self, data_path, ch, anydepth, augmentation, input_size=(512, 512)):
        super(DataSingle, self).__init__()
        self.image_list = self.get_image_list(data_path)
        self.channel = ch
        self.anydepth = anydepth
        self.augmentation = augmentation
        #self.Counter = 0
        if self.augmentation:
            
            # Define augmentation pipeline IMGAUG.
            self.transforms_dict = {
                tio.transforms.RandomAffine(scales=(0.9, 1.2), degrees=40): 0.1,
                tio.transforms.RandomElasticDeformation(num_control_points=7, locked_borders=2): 0.1,
                tio.transforms.RandomAnisotropy(axes=(1, 2), downsampling=(2, 4)): 0.1,
                tio.transforms.RandomBlur(): 0.1,
                tio.transforms.RandomGhosting(): 0.1,
                tio.transforms.RandomSpike(num_spikes = 1, intensity= (1, 2)): 0.1,
                tio.transforms.RandomBiasField(coefficients = 0.2, order= 3): 0.1,
                tio.RandomGamma(log_gamma=0.1): 0.1,
            }
        self.height = input_size[0]
        self.width = input_size[1]
        # self.transform = transforms.Compose(
        #                            [RandomGenerator(output_size=[input_size[0], input_size[1]])])
        # self.normalizeTorch = transforms.Compose([
        # transforms.ToTensor(),
        # transforms.Normalize([0.5], [0.5])
        # ])

    def transform(self, sample):
        image, label = sample['image'], sample['label']
        #self.Counter +=1
        if self.augmentation:
            if random.random() > 0.5:
                sample = RadiologyAugmentationTIO(sample, self.transforms_dict)
            image, label = sample['image'], sample['label']
            #     cv2.imwrite(os.path.join('deneme/','torchio'+str(self.Counter)+'.png'),image)
            #     cv2.imwrite(os.path.join('deneme/','torchio'+str(self.Counter)+'_label.png'),label)
                
            
        if len(image.shape)==2:
            y, x = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
        else:
            y, x, c = image.shape
            if x != self.width or y != self.height:
                image = zoom(image, (self.width / x, self.height / y,1), order=3)  # why not 3?
                label = zoom(label, (self.width / x, self.height / y), order=0)
            
        #z normalizization
        mean3d = np.mean(image, axis=(0,1))
        std3d = np.std(image, axis=(0,1))
        image = (image-mean3d)/std3d
        if len(image.shape)==2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  
            # HWC to CHW, BGR to RGB (for three channel)
            image = image.transpose((2, 0, 1))[::-1]
            image = torch.from_numpy(image.astype(np.float32))

        #image = self.normalizeTorch(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        if self.channel==1:
            if self.anydepth: 
                image = cv2.imread(img_path, -1)
            else:
                image = cv2.imread(img_path, 0)
        elif self.channel==3:
            image = cv2.imread(img_path)

        label_path =  img_path[:img_path.rfind('.')] + '_label.png'
        label = cv2.imread(label_path, 0)
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)

        return sample['image'], sample['label']

    def __len__(self):
        return len(self.image_list)

    def natural_sort(self, l):
        def convert(text): return int(text) if text.isdigit() else text.lower()
        def alphanum_key(key): return [convert(c)
                                       for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def get_image_list(self, path):
        image_paths = []
        for current_path in path:
            for maindir, subdir, file_name_list in os.walk(current_path):
                for filename in file_name_list:
                    if '_label' in filename:
                        continue
                    apath = os.path.join(maindir, filename)
                    ext = os.path.splitext(apath)[1]
                    if ext in image_ext:
                        image_paths.append(apath)
        return self.natural_sort(image_paths)
