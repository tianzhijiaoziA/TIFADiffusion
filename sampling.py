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
# pytype: skip-file
"""Various sampling methods."""
import functools
import time

import torch
import numpy as np
import abc
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet, denoise_tv_bregman
import matplotlib.pyplot as plt
import functools
from utils import fft2, ifft2, clear, fft2_m, ifft2_m, root_sum_of_squares
from tqdm import tqdm
from models import utils as mutils
from models.utils import get_score_fn
#from skimage.measure import compare_psnr,compare_ssim
_CORRECTORS = {}
_PREDICTORS = {}
import pydicom as dicom 
#import odl
#import sde_lib
#from scipy import integrate
from cv2 import imwrite,resize
from utils import get_data_scaler, get_data_inverse_scaler
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse
###
import sys
import time
import threading
import datetime
import os
 
class Logger(object):
    def __init__(self, logf):
        self.logf = logf
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(self.logf, 'a+')
        self.previousMsg = None
 
    def write(self, message):
        if self.previousMsg == None or "\n" in self.previousMsg:
            topMsg = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") + " : "
            self.terminal.write(topMsg)
            self.log.write(topMsg)
        
        if isinstance(message, str):
            self.previousMsg = message
        if self.previousMsg == None:
            self.previousMsg = ""
        
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        self.log.flush()
            
    
def threading_log(log, wait=5):
    while 1== 1:
        time.sleep(wait)
        log.flush()
 
def start_log(logf="logs/temp.log"):
    if not os.path.exists(os.path.dirname(logf)):
        os.makedirs(os.path.dirname(logf))
    
    logger = Logger(logf)
     
    log_thread = threading.Thread(target=threading_log ,args=(logger,))
    log_thread.start()




def set_predict(num):
  if num == 0:
    return 'None'
  elif num == 1:
    return 'EulerMaruyamaPredictor'
  elif num == 2:
    return 'ReverseDiffusionPredictor'

def set_correct(num):
  if num == 0:
    return 'None'
  elif num == 1:
    return 'LangevinCorrector'
  elif num == 2:
    return 'AnnealedLangevinDynamics'

def padding_img(img):
    b,w,h = img.shape
    h1 = 768
    tmp = np.zeros([b,h1,h1])
    x_start = int((h1 -w)//2)
    y_start = int((h1 -h)//2)
    tmp[:,x_start:x_start+w,y_start:y_start+h] = img
    return tmp

def unpadding_img(img):
    b,w,h = img.shape[0],720,720
    h1 = 768
    x_start = int((h1 -w)//2)
    y_start = int((h1 -h)//2)
    return img[:,x_start:x_start+w,y_start:y_start+h]

def init_ct_op(img,r):
  batch = 1#img.shape[0]
  
  sinogram = np.zeros([batch,720,720])
  sparse_sinogram = np.zeros([batch,720,720])
  ori_img = np.zeros_like(img)
  sinogram_max = np.zeros([1,1])
  sinogram[:,...] = Fan_ray_trafo(img[:,...]).data
  
  ori_img[:,...] = Fan_FBP(sinogram[:,...]).data
  sinogram_max[:,0] = sinogram[:,...].max()
  # sinogram[i,...] /= sinogram_max[i,0]
  t = np.copy(sinogram[:,::r,:])
  sparse_sinogram[:,...] = resize(t,[720,720])
  #ori_img为稀疏视图的图像
    
  
  return ori_img, sparse_sinogram.astype(np.float32), sinogram.astype(np.float32),sinogram_max
########################################
def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  predictor = get_predictor(config.sampling.predictor.lower())
  corrector = get_corrector(config.sampling.corrector.lower())
  sampling_fn = get_pc_sampler(sde=sde,
                               shape=shape,
                               predictor=predictor,
                               corrector=corrector,
                               inverse_scaler=inverse_scaler,
                               snr=config.sampling.snr,
                               n_steps=config.sampling.n_steps_each,
                               probability_flow=config.sampling.probability_flow,
                               continuous=config.training.continuous,
                               denoise=config.sampling.noise_removal,
                               eps=eps,
                               device=config.device)
  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    alpha = torch.ones_like(t)

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean
    
@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    # if not isinstance(sde, sde_lib.VPSDE) \
    #     and not isinstance(sde, sde_lib.VESDE) \
    #     and not isinstance(sde, sde_lib.subVPSDE):
    #   raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    # if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    #   timestep = (t * (sde.N - 1) / sde.T).long()
    #   alpha = sde.alphas.to(t.device)[timestep]
    # else:
    alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]

    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

    return x, x_mean

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  corrector_obj = corrector(sde, score_fn, snr, n_steps)
  fn = corrector_obj.update_fn(x, t)
  return fn


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.

    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = sde.prior_sampling(shape).to(device)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      time_corrector_tot = 0
      time_predictor_tot = 0
      for i in range(sde.N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        tic_corrector = time.time()
        x, x_mean = predictor_update_fn(x, vec_t, model=model)
        time_corrector_tot += time.time() - tic_corrector
        tic_predictor = time.time()
        x, x_mean = corrector_update_fn(x, vec_t, model=model)
        time_predictor_tot += time.time() - tic_predictor
      print(f'Average time for corrector step: {time_corrector_tot / sde.N} sec.')
      print(f'Average time for predictor step: {time_predictor_tot / sde.N} sec.')

      return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler


def get_pc_fouriercs_fast(sde, predictor, corrector, inverse_scaler, snr,
                          n_steps=1, probability_flow=False, continuous=False,
                          denoise=True, eps=1e-5, save_progress=False, save_root=None):
  """Create a PC sampler for solving compressed sensing problems as in MRI reconstruction.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    A CS solver function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def data_fidelity(mask, x, Fy):
      """
      Data fidelity operation for Fourier CS
      x: Current aliased img
      Fy: k-space measurement data (masked)
      """
      x = torch.real(ifft2(fft2(x) * (1. - mask) + Fy))
      x_mean = torch.real(ifft2(fft2(x) * (1. - mask) + Fy))
      return x, x_mean

  def get_fouriercs_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def fouriercs_update_fn(model, data, mask, x, t, Fy=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        x, x_mean = data_fidelity(mask, x, Fy)
        return x, x_mean

    return fouriercs_update_fn

  projector_fouriercs_update_fn = get_fouriercs_update_fn(predictor_update_fn)
  corrector_fouriercs_update_fn = get_fouriercs_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, mask, Fy=None):
    with torch.no_grad():
      # Initial sample
      x = torch.real(ifft2(Fy + fft2(sde.prior_sampling(data.shape).to(data.device)) * (1. - mask)))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in tqdm(range(sde.N), total=sde.N):
        t = timesteps[i]
        x, x_mean = corrector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        x, x_mean = projector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        if save_progress and i >= 300 and i % 100 == 0:
          plt.imsave(save_root / f'step{i}.png', clear(x_mean), cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs


def get_pc_fouriercs_RI(sde, predictor, corrector, inverse_scaler, snr,
                        n_steps=1, probability_flow=False, continuous=False,
                        denoise=True, eps=1e-5):
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)
  def TV_J(real):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from skimage.data import chelsea, hubble_deep_field
    from skimage.metrics import mean_squared_error as mse 
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.restoration import (calibrate_denoiser,
                                     denoise_wavelet,
                                     denoise_tv_chambolle, denoise_nl_means,
                                     estimate_sigma)
    from skimage.util import img_as_float, random_noise
    from skimage.color import rgb2gray
    from functools import partial
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)
    _denoise_tv_chambolle = partial(denoise_tv_chambolle, rescale_sigma=True)




     

    # Parameters to test when calibrating the denoising algorithm
    parameter_ranges001gao = {'sigma': np.linspace(10, 0.01,30),
                        'wavelet': ['sym9', 'haar'], 
                       'mode':['soft'],
                       'wavelet_levels':[4,6],
                       'method':['BayesShrink', 'VisuShrink']}
    parameter_ranges001unform = {'sigma': [0.48,0.46,0.44,0.42,0.4,0.38,0.36,0.34,0.32,0.30,0.28,0.26,0.24,0.22,0.2,0.18,0.167,0.15,0.134,0.12,0.1,0.067,0.034,0.001,0.00067,0.00034,0.0001,0.00005],
                        'wavelet': ['db1', 'db2', 'haar'],
                       'mode':['soft'],
                       'wavelet_levels':[4,6],
                       'method':['BayesShrink', 'VisuShrink']}
    # parameter_rangescham = {'weight': np.linspace(0.1, 0.0001,30),
    #                    'eps':np.linspace(0.001, 0.0001,10),
    #                    'max_num_iter':[20,30,50]

    # Denoised image using default parameters of `denoise_wavelet`
    default_output = denoise_wavelet(real)

    # Calibrate denoiser
    calibrated_denoiser = calibrate_denoiser(real,
                                             _denoise_wavelet,
                                             denoise_parameters=parameter_ranges001gao
                                             )

    # Denoised image using calibrated denoiser
    calibrated_output = calibrated_denoiser(real)
    return calibrated_output
  
  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def data_fidelity(mask, x, x_mean, Fy):
      x = ifft2(fft2(x) * (1. - mask) + Fy)
      x_mean = ifft2(fft2(x_mean) * (1. - mask) + Fy)
      return x, x_mean

  def get_fouriercs_update_fn(update_fn):
    def fouriercs_update_fn(model, data, mask, x, t, Fy=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        #实数部数据、虚数部数据
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)
        #实数+虚数部分形成复杂图像
        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean
        x, x_mean = data_fidelity(mask, x, x_mean, Fy)
        return x, x_mean

    return fouriercs_update_fn

  projector_fouriercs_update_fn = get_fouriercs_update_fn(predictor_update_fn)
  corrector_fouriercs_update_fn = get_fouriercs_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, mask, Fy=None, inputer=None):
    with torch.no_grad():
      # Initial sample (complex-valued)
      
      
      
            
      x = ifft2(Fy + fft2(sde.prior_sampling(data.shape).to(data.device)) * (1. - mask))
      
      
      timesteps = torch.linspace(sde.T, eps, sde.N)
      averpsnr=averssim=max_psnr = 0
      max_ssim = 0
      psnr3 = psnr4 = psnr5 = 0
      for i in tqdm(range(sde.N)):
        
        t = timesteps[i]
        

        # x_mean=x.mean(dim=-3)
        # x_mean = x_mean.squeeze().cpu().detach().numpy()
        # tv = TV(x_mean)
        # tv = np.expand_dims(tv,2)
        # tv = np.tile(tv,(1,1,1)) 
        
        # tv = tv.transpose((2,0,1))
        # tv = np.expand_dims(tv,0)
        # tv = torch.from_numpy(tv.astype(np.float32)).cuda()

        x, x_mean = corrector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        #x_mean = ifft2(fft2(data.to(data.device)) + fft2(x_mean) * (1. - mask))
        x, x_mean = projector_fouriercs_update_fn(model, data, mask, x, t, Fy=Fy)
        #x_mean = ifft2(Fy + fft2(x_mean) * (1. - mask))
        # # if i < 20:
        
        # # ##
        # if ((i+2) % 10 == 0) and (i>50):
          
        #   print("TV")
        #   ########TV
        #   #x_mean=x_mean.mean(dim=-3) 
          
        #   x_real = torch.real(x_mean)
        #   x_imag = torch.imag(x_mean)
          
        #   x_imag = x_imag.squeeze().cpu().detach().numpy()
        #   x_real = x_real.squeeze().cpu().detach().numpy()
          
        #   for j in range(3):
            
        #     x_real = TV_J(x_real)
        #     x_imag = TV_J(x_imag)
        #     #x_real=denoise_tv_chambolle(x_real, weight=50,eps=0.1)
        #     #x_imag=denoise_tv_chambolle(x_imag, weight=50,eps=0.1)
        #   tv_real = TV_J(x_real)
        #   tv_imag = TV_J(x_imag)
        #   #tv_real=denoise_tv_chambolle(x_real, weight=50,eps=0.1)
        #   #tv_imag=denoise_tv_chambolle(x_imag, weight=50,eps=0.1)
        #   #tv_real = denoise_wavelet(x_real, sigma=None, wavelet='haar', mode='soft', wavelet_levels=8)
        #   #tv_imag = denoise_wavelet(x_imag, sigma=None, wavelet='haar', mode='soft', wavelet_levels=8)
        #   tv3 = tv_real + 1j * tv_imag
        #   #x_mean=tv3
        #   #print(np.max(tv3),np.min(tv3),np.max(xjisuan),np.min(xjisuan))
         
              
            
          
          
          
            
           
 
        #   x_mid = np.zeros([1,1,320,320],dtype=np.complex64)
        #   x_rec = np.clip(tv3,0,1)
        #   x_rec = np.expand_dims(x_rec,2)
        #   x_mid_1 = np.tile(x_rec,[1,1,1])
        #   x_mid_1 = np.transpose(x_mid_1,[2,0,1]) 
        #   x_mid[0,:,:,:] = x_mid_1
        #   x_mean = torch.tensor(x_mid,dtype=torch.complex64).cuda()
        # # # # #
        # #########
        from matplotlib import cm
        xer = inverse_scaler(x_mean)
        xer=xer.mean(dim=-3)
        xer = xer.squeeze().cpu().detach().numpy() 
        #xer = xer/np.max(xer)
        psnr1 = compare_psnr(255*np.real(xer),255*np.real(inputer),data_range=255)
        ssim1 = compare_ssim(255*np.real(xer),255*np.real(inputer),multichannel=True,data_range=255)
        if psnr1>max_psnr:
          max_psnr = psnr1
        if ssim1>max_ssim: 
          max_ssim = ssim1
          best_img = x_mean
          # real_linshi=denoise_wavelet(torch.real(best_img).squeeze().cpu().detach().numpy(), sigma=0.01, wavelet='haar', mode='soft', wavelet_levels=4)
          # imag_linshi=denoise_wavelet(torch.imag(best_img).squeeze().cpu().detach().numpy(), sigma=0.01, wavelet='haar', mode='soft', wavelet_levels=4)
          # linshi = real_linshi + 1j * imag_linshi
          # img = torch.from_numpy(linshi)
    
          # best_img = img.view(1, 1, 320, 320).to('cuda')  

        print("Num:",i,"PSNR:%.4f"%(psnr1),"SSIM:%.4f"%(ssim1))
      print("PSNR:%.4f"%(max_psnr),"SSIM:%.4f"%(max_ssim))
      averpsnr=averpsnr+max_psnr
      averssim=averssim+max_ssim
      # with open('/data/wyy/score-MRI-main/results/dataceshinotallr4.txt', 'w') as f:  # 设置文件对象
      #   print("PSNR:%.4f"%(max_psnr),"SSIM:%.4f"%(max_ssim),file = f)
      #   print("tongji",averpsnr/11,averssim/11,file = f)
    return inverse_scaler(best_img if denoise else x)
      

  return pc_fouriercs


def get_pc_fouriercs_RI_PI_SSOS(sde, predictor, corrector, inverse_scaler, snr,
                                n_steps=1, probability_flow=False, continuous=False,
                                denoise=True, eps=1e-5, mask=None,
                                save_progress=False, save_root=None):
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # functions to impose data fidelity 1/2\|Ax - y\|^2
  def data_fidelity(mask, x, x_mean, y):
      x = ifft2_m(fft2_m(x) * (1. - mask) + y)
      x_mean = ifft2_m(fft2_m(x_mean) * (1. - mask) + y)
      return x, x_mean

  def get_coil_update_fn(update_fn):
    def fouriercs_update_fn(model, data, x, t, y=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean

        # coil mask
        mask_c = mask[0, 0, :, :].squeeze()
        x, x_mean = data_fidelity(mask_c, x, x_mean, y)
        return x, x_mean

    return fouriercs_update_fn

  predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
  corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, y=None):
    with torch.no_grad():
      # Initial sample: [1, 15, 320, 320] (dtype: torch.complex64)
      x_r = sde.prior_sampling(data.shape).to(data.device)
      x_i = sde.prior_sampling(data.shape).to(data.device)
      x = torch.complex(x_r, x_i)
      x_mean = x.clone().detach()

      timesteps = torch.linspace(sde.T, eps, sde.N)

      # number of iterations of PC sampler
      for i in tqdm(range(sde.N)):
        # coil x_c update
        for c in range(15):
          t = timesteps[i]
          # slicing the dimension with c:c+1 ("one-element slice") preserves dimension
          x_c = x[:, c:c+1, :, :]
          y_c = y[:, c:c+1, :, :]
          x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
          x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)

          # Assign coil dates to the global x, x_mean
          x[:, c, :, :] = x_c
          x_mean[:, c, :, :] = x_c_mean
        if save_progress:
          if i % 100 == 0:
            for c in range(15):
              x_c = clear(x[:, c:c+1, :, :])
              plt.imsave(save_root / 'recon' / f'coil{c}' / f'after{i}.png', np.abs(x_c), cmap='gray')
            x_rss = clear(root_sum_of_squares(torch.abs(x), dim=1).squeeze())
            plt.imsave(save_root / 'recon' / f'after{i}.png', x_rss, cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs


def get_pc_fouriercs_RI_coil_SENSE(sde, predictor, corrector, inverse_scaler, snr,
                                   n_steps=1, lamb_schedule=None, probability_flow=False, continuous=False,
                                   denoise=True, eps=1e-5, sens=None, mask=None, m_steps=10,
                                   save_progress=False, save_root=None):
  '''Every once in a while during separate coil reconstruction,
  apply SENSE data consistency and incorporate information.
  (Args)
    (sens): sensitivity maps
    (m_steps): frequency in which SENSE operation is incorporated
  '''
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  # functions to impose data fidelity 1/2\|Ax - y\|^2
  def data_fidelity(mask, x, x_mean, y):
      x = ifft2_m(fft2_m(x) * (1. - mask) + y)
      x_mean = ifft2_m(fft2_m(x_mean) * (1. - mask) + y)
      return x, x_mean

  def A(x, sens=sens, mask=mask):
      return mask * fft2_m(sens * x)

  def A_H(x, sens=sens, mask=mask):  # Hermitian transpose
      return torch.sum(torch.conj(sens) * ifft2_m(x * mask), dim=1).unsqueeze(dim=1)

  def kaczmarz(x, x_mean, y, lamb=1.0):
      x = x + lamb * A_H(y - A(x))
      x_mean = x_mean + lamb * A_H(y - A(x_mean))
      return x, x_mean

  def get_coil_update_fn(update_fn):
    def fouriercs_update_fn(model, data, x, t, y=None):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        # split real / imag part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # perform update step with real / imag part seperately
        x_real, x_real_mean = update_fn(x_real, vec_t, model=model)
        x_imag, x_imag_mean = update_fn(x_imag, vec_t, model=model)

        # merge real / imag values to form complex image
        x = x_real + 1j * x_imag
        x_mean = x_real_mean + 1j * x_imag_mean

        # coil mask
        mask_c = mask[0, 0, :, :].squeeze()
        x, x_mean = data_fidelity(mask_c, x, x_mean, y)
        return x, x_mean

    return fouriercs_update_fn

  predictor_coil_update_fn = get_coil_update_fn(predictor_update_fn)
  corrector_coil_update_fn = get_coil_update_fn(corrector_update_fn)

  def pc_fouriercs(model, data, y=None):
    with torch.no_grad():
      # Initial sample: [1, 15, 320, 320] (dtype: torch.complex64)
      x_r = sde.prior_sampling(data.shape).to(data.device)
      x_i = sde.prior_sampling(data.shape).to(data.device)
      x = torch.complex(x_r, x_i)
      x_mean = x.clone().detach()

      timesteps = torch.linspace(sde.T, eps, sde.N)

      # number of iterations of PC sampler
      for i in tqdm(range(sde.N)):
        # coil x_c update
        for c in range(15):
          t = timesteps[i]

          # slicing the dimension with c:c+1 ("one-element slice") preserves dimension
          x_c = x[:, c:c+1, :, :]
          y_c = y[:, c:c+1, :, :]
          x_c, x_c_mean = predictor_coil_update_fn(model, data, x_c, t, y=y_c)
          x_c, x_c_mean = corrector_coil_update_fn(model, data, x_c, t, y=y_c)

          # Assign coil dates to the global x, x_mean
          x[:, c, :, :] = x_c
          x_mean[:, c, :, :] = x_c_mean

        # global x update
        if i % m_steps == 0:
          lamb = lamb_schedule.get_current_lambda(i)
          x, x_mean = kaczmarz(x, x_mean, y, lamb=lamb)
        if save_progress:
          if i % 100 == 0:
            for c in range(15):
              x_c = clear(x[:, c:c+1, :, :])
              plt.imsave(save_root / 'recon' / f'coil{c}' / f'after{i}.png', np.abs(x_c), cmap='gray')
            x_rss = clear(root_sum_of_squares(torch.abs(x), dim=1).squeeze())
            plt.imsave(save_root / 'recon' / f'after{i}.png', x_rss, cmap='gray')

      return inverse_scaler(x_mean if denoise else x)

  return pc_fouriercs
##############################




