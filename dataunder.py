import h5py
import cv2
import numpy as np
import torch
f = h5py.File('file1002571_v2.h5','r')
print(f.keys())
f_reconstruction_esc = f['kspace']
data_reconstruction_esc = f_reconstruction_esc[17,: ,:]
im=np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(data_reconstruction_esc)))
#print(im)
# print(im.shape)
# real = np.real(im)
# imag = np.imag(im)
im = cv2.resize(im,(320,320))
# imag = cv2.resize(imag,(320,320))

#f = np.array(data_reconstruction_esc).astype(np.complex64)

np.save('./file1002571_v2_esc17.npy',im)