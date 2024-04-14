from models import utils as mutils
import torch
import numpy as np
from sampling import shared_corrector_update_fn, shared_predictor_update_fn
import functools
from physics.ct import CT
from utils import clear
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# for radon
from build_gemotry import rec
import gc
import scipy.io as io
from odl.contrib import torch as odl_torch
from mask import generate_mask,rec_image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse
# para_ini = initialization()
# fp, fbp = build_gemotry(para_ini)
# ###GPU的radon
# op_modfp = odl_torch.OperatorModule(fp)
# ###GPU的iradon
# op_modpT = odl_torch.OperatorModule(fbp)
rec_al = rec()
def np2torch(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,256,256)
  x = x.to(x_device.device)
  return x
def np2torch_radon(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,720,640)
  x = x.to(x_device.device)
  return x 
def np2torch_radon_view(x,x_device):
  x = torch.from_numpy(x)
  x = x.view(1,1,312,640)
  x = x.to(x_device.device)
  return x 

def get_pc_radon_MCG(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=False):
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

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def _AINV(sinogram):
        return radon.A_dagger(sinogram)

    def _A_all(x):
        return radon_all.A(x)

    def _AINV_all(sinogram):
        return radon_all.A_dagger(sinogram)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            # mn True
            if measurement_noise:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

            # input to the score function
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)

            lamb = lamb_schedule.get_current_lambda(i)

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
            hatx0 = x + (bt ** 2) * score

            # MCG method
            # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
            norm = torch.norm(_AINV(measurement - _A(hatx0)))
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            norm_grad *= weight
            norm_grad = _AINV_all(_A_all(norm_grad) * (1. - mask))

            x_next = x_next + lamb * _AT(measurement - _A(x_next)) / norm_const - norm_grad
            x_next = x_next.detach()
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        x = sde.prior_sampling(data.shape).to(data.device)

        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones))
        timesteps = torch.linspace(sde.T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                          norm_const=norm_const)
            if save_progress:
                if (i % 100) == 0:
                    plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x), cmap='gray')

        return inverse_scaler(x if denoise else x)

    return pc_radon



def get_pc_radon_DPS(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False, weight=1.0,
                     denoise=True, eps=1e-5, radon=None, radon_all=None, save_progress=False, save_root=None,
                     lamb_schedule=None, mask=None, measurement_noise=False):
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

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def _AINV(sinogram):
        return radon.A_dagger(sinogram)

    def _A_all(x):
        return radon_all.A(x)

    def _AINV_all(sinogram):
        return radon_all.A_dagger(sinogram)

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, _, _ = update_fn(x, vec_t, model=model)
                return x

        return radon_update_fn
    
    
    def DPS_get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            vec_t = torch.ones(data.shape[0], device=data.device) * t

            # mn True
            if True:
                measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]

            # input to the score function
            #需要确认一下是否需要加入x_next_mean
            x = x.requires_grad_()
            x_next, x_next_mean, score = update_fn(x, vec_t, model=model)
            if (i==sde.N or i==(sde.N-1)):
                x_next = x_next_mean
                
            lamb = lamb_schedule.get_current_lambda(i)

            # x0 hat estimation
            _, bt = sde.marginal_prob(x, vec_t)
    
            hatx0 = x + (bt[:, None, None, None] ** 2) * score 
            
            # DPS
            difference = measurement - _A(hatx0)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x)[0]
            # MCG method
            # norm = torch.linalg.norm(_AINV(measurement - _A(hatx0)))
            norm_grad *= weight
            
            
            x_next = x_next - norm_grad
            x_next = x_next.detach()
            
            
            return x_next

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = DPS_get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        x = sde.prior_sampling(data.shape).to(data.device)

        ones = torch.ones_like(x).to(data.device)
        norm_const = _AT(_A(ones))
        datanp = np.zeros([300,4])
        n=0
        timesteps = torch.linspace(sde.T, eps, sde.N)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x = predictor_denoise_update_fn(model, data, x, t)
            x = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                          norm_const=norm_const)
            if True:
                if (i % 50) == 0:
                    plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')
                    for i in range(1):
                        psnr1 = compare_psnr(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                        ssim1 = compare_ssim(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                        print("PSNR and SSIM",psnr1,ssim1)
                        print(int(n/10))
                        datanp[int(n/10),1]=psnr1
                        datanp[int(n/10),2]=ssim1
            
            n=n+1
        return inverse_scaler(x if denoise else x)

    return pc_radon


###############################
##################################

def get_pc_radon_song_abation(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None, lamb=1.0,
                      lamb_schedule=None):
    """ Sparse application of measurement consistency """
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

    def _A(x):
        data = x
        sino = rec_al.fp_view(clear(x))
        
        return np2torch_radon_view(sino,data)
    
    def _A_all(x):
        data = x
        sino = rec_al.fp(clear(x))
        
        return np2torch_radon(sino,data)

    def _A_dagger(sinogram):
        data = sinogram
        fbp = rec_al.fbp(clear(sinogram))
        return  np2torch(fbp,data)
    
    def _AT(sinogram):
        data = sinogram
        bp = rec_al.bp_view(clear(sinogram))
        return np2torch(bp,data)
    
    def kaczmarz(x, x_mean, y, lamb=1.0, norm_const=1.0):
      x = x + lamb * _AT(y - _A(x)) / norm_const
      #x_mean = x_mean + lamb * _AT(y - _A(x_mean)) / norm_const
      return x

    # def data_fidelity(mask, x_start, x_mean_start, vec_t=None, measurement=None, lamb=lamb, i=None):
    #     x = torch.mean(x_start, dim=1).unsqueeze(1)
    #     x_mean = torch.mean(x_mean_start, dim=1).unsqueeze(1)
        
    #     y_mean, std = sde.marginal_prob(measurement, vec_t)
    #     hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
    #     weighted_hat_y = hat_y * lamb

    #     sino = _A(x,x)
    #     sino_meas = sino * mask
    #     weighted_sino_meas = sino_meas * (1 - lamb)
    #     sino_unmeas = sino * (1. - mask)

    #     weighted_sino = weighted_sino_meas + sino_unmeas

    #     updated_y = weighted_sino + weighted_hat_y
    #     x = _A_dagger(updated_y,updated_y)

    #     sino_mean = _A(x_mean,x_mean)
    #     updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
    #     x_mean = _A_dagger(updated_y_mean,updated_y_mean)
    #     x_end = x.repeat(1,2,1,1)
    #     x_mean_end = x_mean.repeat(1,2,1,1)
       
    #     return x_end, x_mean_end
    
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
               
                return x, x_mean

        return radon_update_fn
    
    def get_update_fn_sirt(update_fn):
        def radon_update_fn(model, data, x, t,measurement,mask,flag,parter,noise,numrepeat):
            with torch.no_grad():
                gc.disable()
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                vec_t_noise = torch.ones(measurement.shape[0], device=data.device) * t
                x_a, x_mean_a= update_fn(x, vec_t, model=model)

                if True:
                    Mx = torch.mean(x,dim=0).unsqueeze(0)
                    Mx_mean_a = np2torch(rec_al.SIRT_sparse(clear(Mx),clear(measurement),First=False),data)
                    x_mean_a = Mx_mean_a.repeat(2,1,1,1)
                    
                    x = x_mean_a-parter*(x-x_a)
         
                del x_a,x_mean_a,vec_t
                return x 

        return radon_update_fn
    
    def get_update_fn_sirt_x_mean(update_fn):
        def radon_update_fn(model, data, x, t,measurement,parter,noise,numrepeat):
            with torch.no_grad():
            
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x_a, x_mean_a = update_fn(x, vec_t, model=model)
                vec_t_noise = torch.ones(measurement.shape[0], device=data.device) * t
            
                if True:
                    Mx = torch.mean(x,dim=0).unsqueeze(0)
                    Mx_a = np2torch(rec_al.SIRT_sparse(clear(Mx),clear(measurement),First=False),data)
                    x_a = Mx_a.repeat(2,1,1,1)
                    
                    x_mean = x_a-parter*(x-x_mean_a)
               
              
                
                del x_a,x_mean_a,vec_t
                return x_mean
        return radon_update_fn
 

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    predictor_radon_update_fn = get_update_fn_sirt(predictor_update_fn)
    mean_predictor_radon_update_fn = get_update_fn_sirt_x_mean(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    corrector_radon_update_fn = get_update_fn_sirt(corrector_update_fn)
    mean_corrector_radon_update_fn = get_update_fn_sirt_x_mean(corrector_update_fn)

    def pc_radon(model, data, mask, measurement,parter,noise,numrepeat):
        with torch.no_grad():
            flag = 0
            x = sde.prior_sampling([numrepeat,1,256,256]).to(data.device) 
        
            
            for p in range(numrepeat):
                x[p] = np2torch(rec_al.SIRT_sparse(clear(x[p]),clear(measurement),First=True),data) #(4,1,256,256)
            plt.imsave(str(save_root / 'recon' / f'SIRT_init.png'), clear(x[0,:,:,:]), cmap='gray') 
            print("SIRT")
            print(measurement.shape)
            
            datanp = np.zeros([300,4])
            n=0
            
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                torch.cuda.empty_cache()
                t = timesteps[i]
             
                
              
                for _ in range(1):    
                    
                    if (i+1)==sde.N:
                        x = corrector_radon_update_fn(model, data, x, t, measurement,mask,flag,parter,noise,numrepeat)
                        x = mean_predictor_radon_update_fn(model, data, x, t, measurement,parter,noise,numrepeat)
                    else:
                        x = corrector_radon_update_fn(model, data, x, t, measurement,mask,flag,parter,noise,numrepeat)
                        x = predictor_radon_update_fn(model, data, x, t, measurement,mask,flag,parter,noise,numrepeat)
                #     x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                #     x, x_mean = corrector_denoise_update_fn(model, data, x, t)
                
                
                
                if True:
                    if (i % 50) == 0:
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')
                        for i in range(1):
                            psnr1 = compare_psnr(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                            ssim1 = compare_ssim(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                            print("PSNR and SSIM",psnr1,ssim1)
                            print(int(n/10))
                            datanp[int(n/10),1]=psnr1
                            datanp[int(n/10),2]=ssim1
                n=n+1
         
            np.save(save_root / 'recon' / 'progress.npy', datanp)
            
            return inverse_scaler(x),inverse_scaler(x)

    return pc_radon


def get_pc_radon_song_early(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None, lamb=1.0,
                      freq=10):
    """ Sparse application of measurement consistency """
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

    def _A(x):
        return radon.A(x)

    def _A_dagger(sinogram):
        return radon.A_dagger(sinogram)

    def data_fidelity(mask, x, x_mean, vec_t=None, measurement=None, lamb=lamb, i=None):
        y_mean, std = sde.marginal_prob(measurement, vec_t)
        hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
        weighted_hat_y = hat_y * lamb

        sino = _A(x)
        sino_meas = sino * mask
        weighted_sino_meas = sino_meas * (1 - lamb)
        sino_unmeas = sino * (1. - mask)

        weighted_sino = weighted_sino_meas + sino_unmeas

        updated_y = weighted_sino + weighted_hat_y
        x = _A_dagger(updated_y)
        
        sino = _A(x_mean)
        sino_meas = sino * mask
        weighted_sino_meas = sino_meas * (1 - lamb)
        sino_unmeas = sino * (1. - mask)

        weighted_sino = weighted_sino_meas + sino_unmeas

        updated_y_mean = weighted_sino + weighted_hat_y
        x_mean = _A_dagger(updated_y_mean)

        # sino_mean = _A(x_mean)
        # updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
        # x_mean = _A_dagger(updated_y_mean)
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _= update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, mask, x, t, measurement=None, i=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                x, x_mean = data_fidelity(mask, x, x_mean, vec_t=vec_t, measurement=measurement, lamb=lamb, i=i)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, mask, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                if (i % 10) == 0:
                    x, x_mean = corrector_radon_update_fn(model, data, mask, x, t, measurement=measurement, i=i)
                else:
                    x, x_mean = corrector_denoise_update_fn(model, data, x, t)
                if save_progress:
                    if i%100==0:
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean), cmap='gray')
            return inverse_scaler(x_mean if denoise else x)

    return pc_radon


def get_pc_radon_CGLS(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None, lamb=1.0,
                      lamb_schedule=None):
    """ Sparse application of measurement consistency """
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

    def _A(x):
        data = x
        sino = rec_al.fp_view(clear(x))
        
        return np2torch_radon_view(sino,data)
    
    def _A_all(x):
        data = x
        sino = rec_al.fp(clear(x))
        
        return np2torch_radon(sino,data)

    def _A_dagger(sinogram):
        data = sinogram
        fbp = rec_al.fbp(clear(sinogram))
        return  np2torch(fbp,data)
    
    def _AT(sinogram):
        data = sinogram
        bp = rec_al.bp_view(clear(sinogram))
        return np2torch(bp,data)
    
    def kaczmarz(x, x_mean, y, lamb=1.0, norm_const=1.0):
      x = x + lamb * _AT(y - _A(x)) / norm_const
      #x_mean = x_mean + lamb * _AT(y - _A(x_mean)) / norm_const
      return x

    # def data_fidelity(mask, x_start, x_mean_start, vec_t=None, measurement=None, lamb=lamb, i=None):
    #     x = torch.mean(x_start, dim=1).unsqueeze(1)
    #     x_mean = torch.mean(x_mean_start, dim=1).unsqueeze(1)
        
    #     y_mean, std = sde.marginal_prob(measurement, vec_t)
    #     hat_y = (y_mean + torch.rand_like(y_mean) * std[:, None, None, None]) * mask
    #     weighted_hat_y = hat_y * lamb

    #     sino = _A(x,x)
    #     sino_meas = sino * mask
    #     weighted_sino_meas = sino_meas * (1 - lamb)
    #     sino_unmeas = sino * (1. - mask)

    #     weighted_sino = weighted_sino_meas + sino_unmeas

    #     updated_y = weighted_sino + weighted_hat_y
    #     x = _A_dagger(updated_y,updated_y)

    #     sino_mean = _A(x_mean,x_mean)
    #     updated_y_mean = sino_mean * mask * (1. - lamb) + sino * (1. - mask) + y_mean * lamb
    #     x_mean = _A_dagger(updated_y_mean,updated_y_mean)
    #     x_end = x.repeat(1,2,1,1)
    #     x_mean_end = x_mean.repeat(1,2,1,1)
       
    #     return x_end, x_mean_end
    
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
               
                return x, x_mean

        return radon_update_fn
    
    def get_update_fn_sirt(update_fn):
        def radon_update_fn(model, data, x, t,measurement,mask,flag,parter,noise):
            with torch.no_grad():
                gc.disable()
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                vec_t_noise = torch.ones(measurement.shape[0], device=data.device) * t
                x_a, x_mean_a = update_fn(x, vec_t, model=model)
                
                
                if noise:
                    y_mean, std = sde.marginal_prob(measurement, vec_t_noise)
                    measurement = y_mean + 0.5*torch.rand_like(y_mean) * std[:, None, None, None]
             
                for p in range(4):
                    x_mean_a[p] = np2torch(rec_al.CGLS_view(clear(x[p]),clear(measurement),First=False),data) #(4,1,256,256)
                # x = np2torch(rec_al.SIRT_view(clear(x),clear(measurement),First=False),measurement)-0.5*(x-x_a)
                x = x_mean_a-parter*(x-x_a)
                del x_a,x_mean_a,vec_t
                return x #(4,1,256,256)

        return radon_update_fn
    
    def get_update_fn_sirt_x_mean(update_fn):
        def radon_update_fn(model, data, x, t,measurement,parter,noise):
            with torch.no_grad():
            
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x_a, x_mean_a = update_fn(x, vec_t, model=model)
                vec_t_noise = torch.ones(measurement.shape[0], device=data.device) * t
                if noise:
                    y_mean, std = sde.marginal_prob(measurement, vec_t_noise)
                    measurement = y_mean + 0.5*torch.rand_like(y_mean) * std[:, None, None, None]
              
                for p in range(4):
                    x_a[p] = np2torch(rec_al.CGLS_view(clear(x[p]),clear(measurement),First=False),data) #(4,1,256,256)
                    # x = np2torch(rec_al.SIRT_view(clear(x),clear(measurement),First=False),measurement)-0.5*(x-x_a)
                x_mean = x_a-parter*(x-x_mean_a)
                del x_a,x_mean_a,vec_t
                return x_mean
        return radon_update_fn
 

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    predictor_radon_update_fn = get_update_fn_sirt(predictor_update_fn)
    mean_predictor_radon_update_fn = get_update_fn_sirt_x_mean(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    corrector_radon_update_fn = get_update_fn_sirt(corrector_update_fn)
    mean_corrector_radon_update_fn = get_update_fn_sirt_x_mean(corrector_update_fn)

    def pc_radon(model, data, mask, measurement,parter,noise):
        with torch.no_grad():
            flag = 0
            x = sde.prior_sampling([4,1,256,256]).to(data.device)  
            #x = generate_mask(x_512)
            
            for p in range(4):
                x[p] = np2torch(rec_al.CGLS_view(clear(x[p]),clear(measurement),First=True),data) #(4,1,256,256)
            plt.imsave(str(save_root / 'recon' / f'SIRT_init.png'), clear(x[0,:,:,:]), cmap='gray') 
            print("SIRT")
            print(measurement.shape)
            
            # ones = torch.ones_like(x).to(data.device)
            # norm_const = _AT(_A(ones))
            datanp = np.zeros([300,4])
    
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                torch.cuda.empty_cache()
                t = timesteps[i]
         
                
              
                for _ in range(1):    
                    
                    if (i+1)==sde.N:
                        x = mean_corrector_radon_update_fn(model, data, x, t, measurement,parter,noise)
                        x = mean_predictor_radon_update_fn(model, data, x, t, measurement,parter,noise)
                    else:
                        x = corrector_radon_update_fn(model, data, x, t, measurement,mask,flag,parter,noise)
                        x = predictor_radon_update_fn(model, data, x, t, measurement,mask,flag,parter,noise)
                #     x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                #     x, x_mean = corrector_denoise_update_fn(model, data, x, t)
                
                
                
                if True:
                    if (i % 10) == 0:
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x[0,:,:,:].unsqueeze(0)), cmap='gray')
                        for i in range(4):
                            psnr1 = compare_psnr(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                            ssim1 = compare_ssim(255*clear(x[i,:,:,:].unsqueeze(0)),255*(clear(data[0,:,:,:].unsqueeze(0))),data_range=256)
                            print("PSNR and SSIM",psnr1,ssim1)
                            datanp[int(i/10),1]=psnr1
                            datanp[int(i/10),2]=ssim1
                            
                        #plt.imsave(save_root / 'recon' / f'progress{i}.png', np.mean(clear(x_mean),axis=0), cmap='gray')
            # #finally
            if False:
                ones = torch.ones_like(x[0].unsqueeze(0)).to(data.device)
                norm_const = _AT(_A(ones))
                for p in range(4):
                    x[p] = kaczmarz(x[p], x, measurement, lamb=1.0, norm_const=norm_const)
            x_512 = rec_image(x[0,:,:,:].unsqueeze(0),x[1,:,:,:].unsqueeze(0),x[2,:,:,:].unsqueeze(0),x[3,:,:,:].unsqueeze(0))
            np.save(save_root / 'recon' / 'our.npy', datanp)
            #plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_512), cmap='gray')
            return inverse_scaler(x),inverse_scaler(x_512)

    return pc_radon


def get_pc_radon_POCS(sde, predictor, corrector, inverse_scaler, snr,
                      n_steps=1, probability_flow=False, continuous=False,
                      denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                      lamb_schedule=None, measurement_noise=False, final_consistency=False):
    """ Sparse application of measurement consistency """
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

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None,
                 norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn

    def get_corrector_update_fn(update_fn):
        def radon_update_fn(model, data, x, t, measurement=None, i=None, norm_const=None):
            with torch.no_grad():
                vec_t = torch.ones(data.shape[0], device=data.device) * t
                x, x_mean, _ = update_fn(x, vec_t, model=model)

                if measurement_noise:
                    measurement_mean, std = sde.marginal_prob(measurement, vec_t)
                    measurement = measurement_mean + torch.randn_like(measurement) * std[:, None, None, None]
                #确定一下lamb的位置，不确定是不是写这里    
                lamb = 10*lamb_schedule.get_current_lambda(i)
                x, x_mean = kaczmarz(x, x_mean, measurement=measurement, lamb=lamb, i=i,
                                     norm_const=norm_const)
                return x, x_mean

        return radon_update_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_radon_update_fn = get_corrector_update_fn(corrector_update_fn)

    def pc_radon(model, data, measurement=None):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                x, x_mean = predictor_denoise_update_fn(model, data, x, t)
                x, x_mean = corrector_radon_update_fn(model, data, x, t, measurement=measurement, i=i,
                                                      norm_const=norm_const)
                if save_progress:
                    if (i % 100) == 0:
                        #print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / f'progress{i}.png', clear(x_mean), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x_mean, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon



