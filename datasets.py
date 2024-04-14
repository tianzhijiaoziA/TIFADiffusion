# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as scio
import pydicom as dicom
import os
class GetCT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])
        #self.data_names = random.shuffle(self.data_names)
        #print(self.data_names)
        self.augment = None

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']

        dataCT=dicom.read_file(self.data_names[index])
        data_array = dataCT.pixel_array.astype(np.float32) * dataCT.RescaleSlope + dataCT.RescaleIntercept
        data_array = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array)) ### 0-1
        data_array = np.expand_dims(data_array,2)
        #data_array_10 = data_array.repeat([1,1,10],axis=2)
        data_array_10 = np.tile(data_array,(1,1,10))
        data_array_10=data_array_10.transpose((2,0,1))
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)


class GetCT_4(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])
        #self.data_names = random.shuffle(self.data_names)
        #print(self.data_names)
        self.augment = None

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']

        dataCT=dicom.read_file(self.data_names[index])
        data_array = dataCT.pixel_array.astype(np.float32) * dataCT.RescaleSlope + dataCT.RescaleIntercept
        data_array = data_array/2500
        data_array = np.expand_dims(data_array,2)
        #data_array_10 = data_array.repeat([1,1,10],axis=2)
        data_array_10 = np.tile(data_array,(1,1,1))
        data_array_10=data_array_10.transpose((2,0,1))
        return data_array_10
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)
          
          
class AAPM(Dataset):

    def __init__(self,root):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

   

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT=scio.loadmat(self.data_names[index])['sub_label']
        #label=scio.loadmat(self.data_names[index])['gt']
        
        data_array = dataCT.astype(np.float32)
        # if np.max(data_array)>10:
        #   data_array = data_array/2500 #估计AAPM三维体的数值范围，做的一致性归一化
          
        data_array = np.expand_dims(data_array,0)
        # data_array_2 = data_array.transpose((1,2,0))
        # #print(data_array.shape)
        # img = resize_small(data_array_2, 256)
        # img_np = np.array(img)
        # imgs = img_np.transpose((2,0,1))
        imgs = data_array
        print(imgs.shape)
        return imgs
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)

class AAPM_4(Dataset):

    def __init__(self,root):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

   

    def __getitem__(self,index):
        #x=loadmat(self.data_names[index])['Img2']
        #data_array = np.load(self.data_names[index]).astype(np.float32)
        dataCT1=np.expand_dims((scio.loadmat(self.data_names[index])['sub_label1']).astype(np.float32),0)
        dataCT2=np.expand_dims((scio.loadmat(self.data_names[index])['sub_label2']).astype(np.float32),0)
        dataCT3=np.expand_dims((scio.loadmat(self.data_names[index])['sub_label3']).astype(np.float32),0)
        dataCT4=np.expand_dims((scio.loadmat(self.data_names[index])['sub_label4']).astype(np.float32),0)
        data_array=np.concatenate([dataCT1,dataCT2,dataCT3,dataCT4],axis=0)
        # if np.max(data_array)>10:
        #   data_array = data_array/2500 #估计AAPM三维体的数值范围，做的一致性归一化
          
        # data_array = np.expand_dims(data_array,0)
        # data_array_2 = data_array.transpose((1,2,0))
        # #print(data_array.shape)
        # img = resize_small(data_array_2, 256)
        # img_np = np.array(img)
        # imgs = img_np.transpose((2,0,1))
        imgs = data_array
        print(imgs.shape)
        return imgs
    
    def __len__(self):
        #if type(self.data_names) != 'str':
            #self.data_names = str(self.data_names)
            return len(self.data_names)


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    #定义一个x的数值
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      # Added to train grayscale models
      # img = tf.image.rgb_to_grayscale(img)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)


  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder


from pathlib import Path
from utils import fft2, ifft2, get_mask
import torch
class fastmri_knee(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, is_complex=False):
    self.root = root
    #self.data_list = list(root.glob('*/*.npy'))
    print(root)
    import glob
    self.data_list = list(glob.glob("/home/wangyy/score_mri/configs/npy/reconstruction_esc/*.npy"))
  
    self.is_complex = is_complex

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname).astype(np.complex64)
    data = (data-np.min(data))/(np.max(data)-np.min(data)) 
    # data = np.load(fname).astype(np.complex64)
    # data = (data-np.min(data))/(np.max(data)-np.min(data))  
    # data = torch.from_numpy(data).view(1, 1, 320, 320) 
    # mask = get_mask(data, 320, 1,
    #               type='gaussian1d',
    #               acc_factor=4,
    #               center_fraction=0.08)
    # # fft.astype(np.complex64)
    # kspace = fft2(data)
    print('full')
    # # undersampling
    # under_kspace = kspace * mask
    # under_img = ifft2(under_kspace)
    # under_img=under_img.mean(dim=-3) 
    # data = under_img.squeeze().cpu().detach().numpy()
  
    
    #print("11111111",data.shape,data[:5,:5]) 
    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)
    return data


class fastmri_knee_infer(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, sort=True, is_complex=False):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))
    self.is_complex = is_complex
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    if not self.is_complex:
      data = np.load(fname)
    else:
      data = np.load(fname).astype(np.complex64)
    data = np.expand_dims(data, axis=0)
    return data, str(fname)


class fastmri_knee_magpha(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname).astype(np.float32)
    return data


class fastmri_knee_magpha_infer(Dataset):
  """ Simple pytorch dataset for fastmri knee singlecoil dataset """
  def __init__(self, root, sort=True):
    self.root = root
    self.data_list = list(root.glob('*/*.npy'))
    if sort:
      self.data_list = sorted(self.data_list)

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, idx):
    fname = self.data_list[idx]
    data = np.load(fname).astype(np.float32)
    return data, str(fname)


def create_dataloader(configs, evaluation=False, sort=True):
  #shuffle = True if not evaluation else False
  shuffle=False
  # if configs.data.is_multi:
  #   train_dataset = fastmri_knee(Path(configs.data.root) / f'knee_multicoil_{configs.data.image_size}_train')
  #   val_dataset = fastmri_knee_infer(Path(configs.data.root) / f'knee_{configs.data.image_size}_val', sort=sort)
  # elif configs.data.is_complex:
  #   if configs.data.magpha:
  #     train_dataset = fastmri_knee_magpha(Path(configs.data.root) / f'knee_complex_magpha_{configs.data.image_size}_train')
  #     val_dataset = fastmri_knee_magpha_infer(Path(configs.data.root) / f'knee_complex_magpha_{configs.data.image_size}_val')
  #   else:
  #     train_dataset = fastmri_knee(root= "/home/wangyy/score_mri/npy/reconstruction_esc")
  #     val_dataset = fastmri_knee(root= "home/wangyy/score_mri/npy/reconstruction_esc")
  # else:
  train_dataset = AAPM_4(root= "/data/wyy/EASEL/EASEL/quarter_1mm/score_special_train_8_concat/")
  val_dataset = AAPM_4(root= "/data/wyy/EASEL/EASEL/quarter_1mm/score_special_train_8_concat/")
    #train_dataset = GetCT(root= "/data/wyy/EASEL/EASEL/quarter_1mm/Ltrain",augment=None)
    
    #val_dataset = GetCT(root= "/data/wyy/EASEL/EASEL/quarter_1mm/full_1mm",augment=None)

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.training.batch_size,
    shuffle=shuffle,
    drop_last=True,
    #worker_init_fn= 123,
  )

  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=configs.training.batch_size,
    # shuffle=False,
    shuffle=True,
    drop_last=True
  )
  return train_loader, val_loader


def create_dataloader_regression(configs, evaluation=False):
  shuffle = True if not evaluation else False
  train_dataset = fastmri_knee(Path(configs.root) / f'knee_{configs.image_size}_train')
  val_dataset = fastmri_knee_infer(Path(configs.root) / f'knee_{configs.image_size}_val')

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=configs.batch_size,
    shuffle=shuffle,
    drop_last=True
  )
  
  val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=configs.batch_size,
    shuffle=False,
    drop_last=True
  )
  return train_loader, val_loader

