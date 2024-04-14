import odl
import numpy as np
import astra


## 360geo
class initialization:
    def __init__(self):
        self.param = {}
        self.reso = 512 * 0.03

        # image
        self.param['nx_h'] = 512
        self.param['ny_h'] = 512
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        self.param['nProj'] = 360

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = 640
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso

        self.param['u_water'] = 0.192


def build_gemotry(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')

    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
    fbp = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Hann', frequency_scaling=0.8)
    
    return ray_trafo_hh, fbp

class rec():
  def __init__(self):
    size = 256
    self.reso = size*0.03
    self.angles = np.linspace(0, np.pi, num=720, endpoint=False)
    self.angles_s = np.linspace(0, np.pi/2, num=360, endpoint=False)
    self.angles_view = np.linspace(0, np.pi, num=23, endpoint=False)
    #有限角度指的是np.pi是有限的个数 90个就是正弦图取一半,不是简单的线性相加
    self.proj_geom = astra.create_proj_geom('fanflat',1.5,640,self.angles,1500,500)
    self.proj_geom_s = astra.create_proj_geom('fanflat',1.5,640,self.angles_s,1500,500)
    # self.proj_geom_s = astra.create_proj_geom('fanflat',0.4,640,self.angles_s,1500,500)
    self.proj_geom_view = astra.create_proj_geom('fanflat',1.5,640,self.angles_view,1500,500)
    #1是pixel，640是角度数，1100，500是SDD和,volpixel的计算通过pixel*探测器个数
    self.vol_geom = astra.create_vol_geom(size,size,int((-size / 2) ),int((size / 2) ),int((-size / 2)),int((size / 2)))
    # self.vol_geom = astra.create_vol_geom(size, size, (-size / 2) * 81 / size, (size / 2) * 81 / size,
    #                                          (-size / 2) * 81 / size, (size / 2) * 81 / size)

  def fp(self, x):
    rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom)
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(proj_id)
  
  def fp_sparse(self, x):
    rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom_s)
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(proj_id)
  
  def fp_view(self, x):
    rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom_view)
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(proj_id)
  
  def fbp(self,x):
    rec_id = astra.data2d.create('-vol',self.vol_geom)
    proj_id = astra.data2d.create('-sino',self.proj_geom,x)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(rec_id)
  
  def fbp_sparse(self,x):
    rec_id = astra.data2d.create('-vol',self.vol_geom)
    proj_id = astra.data2d.create('-sino',self.proj_geom_s,x)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(rec_id)
  
  
  def fbp_view(self,x):
    rec_id = astra.data2d.create('-vol',self.vol_geom)
    proj_id = astra.data2d.create('-sino',self.proj_geom_view,x)
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(rec_id)
  ########bp_view确认一下ReconstructionDataId还是VolumeDataId
  def bp_view(self,x):
    rec_id = astra.data2d.create('-vol',self.vol_geom)
    proj_id = astra.data2d.create('-sino',self.proj_geom_view,x)
    cfg = astra.astra_dict('BP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    return astra.data2d.get(rec_id)
  
  def SIRT(self,x=None,Fy=None,First=False):
    if First:
        rec_id = astra.data2d.create('-vol',self.vol_geom)
    else:
        rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom,Fy)
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id,5)
    return astra.data2d.get(rec_id)
  
  def SIRT_sparse(self,x=None,Fy=None,First=False):
    if First:
        rec_id = astra.data2d.create('-vol',self.vol_geom)
    else:
        rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom_s,Fy)
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id,40)
    return astra.data2d.get(rec_id)
  
  
  def SIRT_view(self,x=None,Fy=None,First=False):
    if First:
        rec_id = astra.data2d.create('-vol',self.vol_geom)
    else:
        rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom_view,Fy)
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id,20)
    xxx = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)
    return xxx
  
  def CGLS_view(self,x=None,Fy=None,First=False):
    if First:
        rec_id = astra.data2d.create('-vol',self.vol_geom)
    else:
        rec_id = astra.data2d.create('-vol',self.vol_geom,x)
    proj_id = astra.data2d.create('-sino',self.proj_geom_view,Fy)
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id,20)
    xxx = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(proj_id)
    return xxx

