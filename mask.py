import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from scipy import io

imgs = torch.ones((1,1, 512, 512), dtype=torch.float).cuda()
shape = imgs.shape                                           #(1,1,256,256)
listimgH = []
Zshape = [shape[0], shape[1],shape[2], shape[3]]            #(1,1,256,256)
imgZ = imgs[:,:, :Zshape[2], :Zshape[3]].cuda()



mask1 = torch.zeros((Zshape[0],Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()               #1,1,256,256
mask2 = torch.zeros((Zshape[0],Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()
mask3 = torch.zeros((Zshape[0], Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()
mask4 = torch.zeros((Zshape[0], Zshape[1], Zshape[2], Zshape[3]), dtype=torch.float).cuda()

for i in range(0,256,2):
    for j in range(0,256,2):

            mask1[:, :, i*2, j*2] = imgZ[:, :, i*2, j*2]
            mask1[:, :, i*2+1, j*2+2] = imgZ[:, :, i*2+1, j*2+2]
            mask1[:, :, i*2+2, j*2] = imgZ[:, :, i*2+2, j*2]
            mask1[:, :, i*2+3, j*2+2] = imgZ[:, :, i*2+3, j*2+2]

            mask2[:, :, i*2, j*2+1] = imgZ[:, :, i*2, j*2+1]
            mask2[:, :, i*2+1, j*2+3] = imgZ[:, :, i*2+1, j*2+3]
            mask2[:, :, i*2+2, j*2+1] = imgZ[:, :, i*2+2, j*2+1]
            mask2[:, :, i*2+3, j*2+3] = imgZ[:, :, i*2+3, j*2+3]

            mask3[:, :, i*2+1, j*2] = imgZ[:, :, i*2+1, j*2]
            mask3[:, :, i*2, j*2+2] = imgZ[:, :, i*2 , j*2+2]
            mask3[:, :, i*2+3, j*2] = imgZ[:, :, i*2+3, j*2]
            mask3[:, :, i*2+2, j*2+2] = imgZ[:, :, i*2+2, j*2+2]

            mask4[:, :, i*2+1, j*2+1] = imgZ[:, :, i*2+1, j*2+1]
            mask4[:, :, i*2, j*2+3] = imgZ[:, :, i*2 , j*2+3]
            mask4[:, :, i*2+3, j*2+1] = imgZ[:, :, i*2+3, j*2+1]
            mask4[:, :, i*2+2, j*2+3] = imgZ[:, :, i*2+2, j*2+3]


def generate_mask(img):#256*256
    noisy=img
    


    # listimgH.append(mask1)
    # listimgH.append(mask2)
    # listimgH.append(mask3)
    # listimgH.append(mask4)
                      
    avgpool=torch.nn.AvgPool2d(2)
    noisy_sub1 = 4 * avgpool(noisy * mask1)
    noisy_sub2 = 4 * avgpool(noisy * mask2)
    noisy_sub3 = 4 * avgpool(noisy * mask3)
    noisy_sub4 = 4 * avgpool(noisy * mask4)
    
    return torch.cat((noisy_sub1,noisy_sub2,noisy_sub3,noisy_sub4),dim=0)


def rec_image(sub1,sub2,sub3,sub4):
    img=torch.zeros([1,1,512,512])
    for i in range(0,256,2):
        for j in range(0,256,2):
            img[:, :, i*2, j*2] = sub1[:, :, i, j]
            img[:, :, i*2+1, j*2+2] = sub1[:, :, i, j+1]
            img[:, :, i*2+2, j*2] = sub1[:, :, i+1, j]
            img[:, :, i*2+3, j*2+2] = sub1[:, :, i+1, j+1]
            
            img[:, :, i*2, j*2+1] = sub2[:, :, i, j]
            img[:, :, i*2+1, j*2+3] = sub2[:, :, i, j+1]
            img[:, :, i*2+2, j*2+1] = sub2[:, :, i+1, j]
            img[:, :, i*2+3, j*2+3] = sub2[:, :, i+1, j+1]
            
            img[:, :, i*2+1, j*2] = sub3[:, :, i, j]
            img[:, :, i*2, j*2+2] = sub3[:, :, i, j+1]
            img[:, :, i*2+3, j*2] = sub3[:, :, i+1, j]
            img[:, :, i*2+2, j*2+2] = sub3[:, :, i+1, j+1]
            
            img[:, :, i*2+1, j*2+1] = sub4[:, :, i, j]
            img[:, :, i*2, j*2+3] = sub4[:, :, i, j+1]
            img[:, :, i*2+3, j*2+1] = sub4[:, :, i+1, j]
            img[:, :, i*2+2, j*2+3] = sub4[:, :, i+1, j+1]

    return img
# #
# #start
# dataset = dicom.read_file('./L067_FD_0399.IMA')
# print(np.max(dataset.pixel_array.astype(np.float32)),np.min(dataset.pixel_array.astype(np.float32)))
# img1 = dataset.pixel_array.astype(np.float32)/2500
# img1 = torch.from_numpy(img1).view(1, 1, 512, 512).cuda()

# one = torch.ones((1,1, 512, 512), dtype=torch.float).cuda()
# noisy = img1
# MS = generate_mask(one)
# mask1 = MS[0]
# mask2 = MS[1]
# mask3 = MS[2]
# mask4 = MS[3]                        
# avgpool=torch.nn.AvgPool2d(2)
# noisy_sub1 = 4 * avgpool(noisy * mask1)
# noisy_sub2 = 4 * avgpool(noisy * mask2)
# noisy_sub3 = 4 * avgpool(noisy * mask3)
# noisy_sub4 = 4 * avgpool(noisy * mask4)
# plt.imsave('noisy.png', noisy.squeeze().cpu().detach().numpy(), cmap='gray')
# plt.imsave('noisy_sub1.png', noisy_sub1.squeeze().cpu().detach().numpy(), cmap='gray')
# plt.imsave('noisy_sub2.png', noisy_sub1.squeeze().cpu().detach().numpy(), cmap='gray')
# plt.imsave('noisy_sub3.png', noisy_sub1.squeeze().cpu().detach().numpy(), cmap='gray')
# plt.imsave('noisy_sub4.png', noisy_sub1.squeeze().cpu().detach().numpy(), cmap='gray')

# input = rec_image(noisy_sub1,noisy_sub2,noisy_sub3,noisy_sub4)
# plt.imsave('input.png', input.squeeze().cpu().detach().numpy(), cmap='gray')
# io.savemat('L607399_down.mat',{'label': img1.squeeze().cpu().detach().numpy(),'sub1': noisy_sub1.squeeze().cpu().detach().numpy(),'sub2': noisy_sub2.squeeze().cpu().detach().numpy(),'sub3': noisy_sub3.squeeze().cpu().detach().numpy(),'sub4': noisy_sub4.squeeze().cpu().detach().numpy()})  
###############################
###############################
# dataset = dicom.read_file('./L067_FD_0399.IMA')
# print(np.max(dataset.pixel_array.astype(np.float32)),np.min(dataset.pixel_array.astype(np.float32)))
# img1 = dataset.pixel_array.astype(np.float32)/2500
# img1 = torch.from_numpy(img1).view(1, 1, 512, 512).cuda()


# sub = generate_mask(img1)


# plt.imsave('noisy_sub1.png', (sub[0,:,:,:].unsqueeze(0)).squeeze().cpu().detach().numpy(), cmap='gray')


# input = rec_image(sub[0,:,:,:].unsqueeze(0),sub[1,:,:,:].unsqueeze(0),sub[2,:,:,:].unsqueeze(0),sub[3,:,:,:].unsqueeze(0))
# plt.imsave('input.png', input.squeeze().cpu().detach().numpy(), cmap='gray')
