import json
import os
import random

import torch
import torch.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from michalski_trains.m_train import *

ATTR_TO_VAL = {
    "length":
    {
        "short": 0,
        "long": 1
    }, 

    "shape":
    {
        "rectangle": 0, #yellow
        "bucket": 1 #green
    }, 
    "roof": 
    {
        "none": 0, 
        "arc": 1
    }, 

    "wall": 
    {
        "not_double": 0, 
        "double": 1
    }, 

    "l_shape": 
    {
        "none": 0, 
        "triangle": 1, 
        "rectangle": 2
    }
}

class CenterCropMultiple(transforms.CenterCrop):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        assert isinstance(pic, list)
        transformed_imgs = []
        for img in pic: 
            transformed_imgs.append(super().__call__(img))
        return transformed_imgs

class ToTensorMultiple(transforms.ToTensor):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        assert isinstance(pic, list)
        transformed_imgs = []
        for img in pic: 
            transformed_imgs.append(super().__call__(img))
        res = torch.stack(transformed_imgs, dim=0)
        return res

class ResizeMultiple(transforms.Resize):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        assert isinstance(pic, torch.Tensor) and len(pic.size()) == 4
        transformed_imgs = []
        for i in range(pic.shape[0]):
            transformed_imgs.append(super().__call__(pic[i]))
        res = torch.stack(transformed_imgs, dim=0)
        return res
    
class NormalizeMultiple(transforms.Normalize):
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        assert isinstance(pic, torch.Tensor) and len(pic.size()) == 4
        transformed_imgs = []
        for i in range(pic.shape[0]):
            transformed_imgs.append(super().__call__(pic[i]))
        res = torch.stack(transformed_imgs, dim=0)
        return res
    


        


class MichalskiIntervenedDataset(Dataset):
    def __init__(self, data_file,
                 ds_size=10000, resize=False, ds_path='output/image_generator',
                 preprocessing=None):
        """ MichalskiTrainDataset
            @param data_file (str): The name of the file containing all data in output/image_generator
            @param ds_size (int): number of train images
            @param resize: bool if true images are resized to 224x224
            @param ds_path: path to the dataset
            @param preprocessing: preprocessing function to apply to the images
            """
        # ds data
        self.images, self.attributes, self.y = [], [], []
        train_vis = data_file.split("_")[0]
        #Attributes in order and their possible values
        self.attribute_classes = {
            "color": ["yellow", "green"], 
            "length": ["short", "long"], 
            "wall": ["not_double", "double"], 
            "roof": ["none", "arc"], 
            "wheels": [2,3], 
            "l_shape": ["none", "triangle", "rectangle"], 
            "l_num": [0,1,2,3]
        }


        
        self.resize, self.train_count, = resize, ds_size
        # ds path
        self.image_base_path = f'{ds_path}/{data_file}/images'
        self.all_scenes_path = f'{ds_path}/{data_file}/all_scenes'

        # ds labels
        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        color = ['yellow', 'green'] #rectangle, triangle
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation"]
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ["box"]
        # train with class specific labels
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError(f'json scene file missing {self.all_scenes_path}. Not all images were generated')
        """ if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}') """

        # load data
        path = self.all_scenes_path + '/all_scenes.json'
        #assumes sorted scene list of interventions
        #orrder is: train_int_length, train_int_shape, train_int_color, train_normal
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            assert len(all_scenes["scenes"]) == ds_size * 4
            for scene_nr in range(ds_size):
                curr_sample = []
                curr_attributes = []
                curr_labels = []
                for i in range(4):
                    scene = all_scenes["scenes"][scene_nr * 4 + i]
                    curr_sample.append(scene["image_filename"])
                    #self.images.append(scene['image_filename'])
                    # self.depths.append(scene['depth_map_filename'])
                    if 'train' in scene:
                        # new json data format
                        train = scene['train']
                        train = MichalskiTrain.from_text(train, train_vis)
                    else:
                        # old json data format
                        train = scene['m_train']
                        train = jsonpickle.decode(train)
                        # self.trains.append(train.replace('michalski_trains.m_train.', 'm_train.'))
                        # text = train.to_txt()
                        # t1 = MichalskiTrain.from_text(text, train_vis)
                    attr = self.__train_to_attribute(train)
                    curr_attributes.append(attr)
                    lab = int(train.get_label() == 'east')
                    curr_labels.append(lab)
                self.y.append(curr_labels)
                self.attributes.append(curr_attributes)
                self.images.append(curr_sample)

        self.y = torch.tensor(self.y).T
        # transforms
        self.image_size = self.get_image_size(0)
        trans = [CenterCropMultiple(270), ToTensorMultiple()]
        if preprocessing is not None:
            trans.append(preprocessing)
        if resize:
            print('resize true')
            #NOTE P.V: changed this because CelebA is 128,128 as well
            trans.append(ResizeMultiple((128, 128), interpolation=transforms.InterpolationMode.BICUBIC))

        trans.append(NormalizeMultiple(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.norm = transforms.Compose(trans)


    def __train_to_attribute(self, train):
        cars = train.get_cars()
        assert len(cars) == 1
        car = cars[0]
        
        print(car.get_car_shape())
        print(car.get_load_shape())
        print("-------------------------")
        color = ATTR_TO_VAL["shape"][car.get_car_shape()]
        length = ATTR_TO_VAL["length"][car.get_car_length()]
        wall = ATTR_TO_VAL["wall"][car.get_car_wall()]
        roof = ATTR_TO_VAL["roof"][car.get_car_roof()]
        #already int
        wheels = car.get_wheel_count()
        l_shape = ATTR_TO_VAL["l_shape"][car.get_load_shape()]
        l_num = car.get_load_number()
        return [length, color, wall, roof, wheels, l_shape, l_num]


    def __getitem__(self, index):
        sample = self.get_pil_image(index)
        X = self.norm(sample)
        attribute = self.attributes[index]
        y = self.get_direction(index)
        return X, attribute, y

    def __len__(self):
        return len(self.y)

    def get_image_size(self, item):
        #single sample
        im = self.get_pil_image(item)[0]
        return im.size

    def get_direction(self, item):
        # lab = self.trains[item].get_label()
        # if lab == 'none':
        #     # return torch.tensor(0).unsqueeze(dim=0)
        #     raise AssertionError(f'There is no direction label for the selected DS {self.ds_typ}')
        # label_binary = self.label_classes.index(lab)
        label_binary = self.y[item]
        label = torch.tensor(label_binary).unsqueeze(dim=0)
        return label

    def apply_label_noise(self, label_noise):
        if label_noise > 0:
            print(f'applying noise of {label_noise} to dataset labels')
            for idx in range(self.__len__()):
                n = random.random()
                if n < label_noise:
                    train = self.get_m_train(idx)
                    self.y[idx] = 1 - self.y[idx]
                    lab = train.get_label()
                    if lab == 'east':
                        train.set_label('west')
                    elif lab == 'west':
                        train.set_label('east')
                    else:
                        raise ValueError(f'unexpected label value {lab}, expected value east or west')

    def apply_image_noise(self, image_noise):
        if image_noise > 0:
            print('adding noise to images')
            self.norm.transforms.insert(-2, AddBinaryNoise(image_noise))

    def get_m_train(self, item):
        return self.trains[item]

    def get_mask(self, item):
        return self.masks[item]

    def get_pil_image(self, item):
        #list
        im_path = self.get_image_path(item)
        assert isinstance(im_path, list)
        return [Image.open(path).convert('RGB') for path in im_path]

    def get_image(self, item):
        im = self.get_pil_image(item)
        return self.norm(im)

    def get_image_path(self, item):
        assert isinstance(self.images[item], list)
        return [self.image_base_path + '/' + self.images[item][i] for i in range(4)] 

    def get_label_for_id(self, item):
        return self.trains[item].get_label()

    def get_trains(self):
        return self.trains

    def get_ds_labels(self):
        return self.labels

    def get_ds_classes(self):
        return self.label_classes

    def get_class_dim(self):
        return len(self.label_classes)

    def get_output_dim(self):
        return len(self.labels)
    
class AddBinaryNoise(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, tensor):
        t = torch.ones_like(tensor)
        t[torch.rand_like(tensor) < self.p] = 0
        return t * tensor

    def __repr__(self):
        return self.__class__.__name__ + '(percentage={0})'.format(self.p)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
