
import numpy as np
import torch
from tqdm import tqdm
import PETLibs.recons.GPURecons as GPURecons
from PETLibs import *

def preprocess_data(image, truncate=104, tensorFlag = False, device="cuda"):
    image = image.squeeze()
    if truncate < image.shape[0]:
        t = image.shape[0] - truncate
        image_sel = image[t:,:,:]
    else:
        image_sel=image
    if tensorFlag:
        image_sel=torch.unsqueeze(torch.unsqueeze(image_sel, 0),0)
    else :
        image_sel=torch.unsqueeze(torch.unsqueeze(torch.tensor(image_sel,device=device, dtype=torch.float32), 0),0)
    return image_sel

def postprocess_data(image, truncate=104, im_dim=[109,128,128], tensorFlag=False):
    image_sel=image
    if tensorFlag:
        image_sel=torch.squeeze(image_sel)
        
        t = im_dim[0] - truncate
        image_rec=torch.zeros(im_dim, device=image_sel.device)
        image_rec[t:,:,:] = image_sel
        return image_rec
       
    else:
        image_sel=np.squeeze(image_sel.detach().cpu().numpy())
       
        t = im_dim[0] - truncate
        image_rec=np.zeros(im_dim)
        image_rec[t:,:,:] = image_sel
        return image_rec

def GPU_log_likelihood(x, eps = 1e-6, BiographParams=None,dict_sino=None):

    BiographReconsOps=GPURecons.GPUReconsOps(BiographParams)
    fwd_func=BiographReconsOps.forwardProj
    nbsubset=max([1,BiographParams.nsubsets])
    x= torch.swapaxes(x,2,0)

    y =  BiographParams.createSubsetData(dict_sino["val"])
    mult_array =  BiographParams.createSubsetData(dict_sino["mult"])
    add_array =  BiographParams.createSubsetData(dict_sino["add"])

    tx=x.contiguous().cuda()
    tx=torch.reshape(tx,BiographParams.tb_im_dim)
    log_l = 0
    for i in range(nbsubset):
        ty=y[i].cuda()
        fwd=fwd_func(tx,mult_factor=mult_array[i],add_factor=add_array[i], subset=i)
        log_l += torch.sum(ty[fwd>eps]*torch.log(fwd[fwd>eps])-fwd[fwd>eps]).item()
    return log_l

def Tensor2Np(tensor):
    return np.squeeze(tensor.detach().cpu().numpy())


def GPUBiographEMRecons(cdh_name,cdf_path,reconsParams=None,verbose=True,xstart=None,
                        dict_sino=None,eval_mode=True,show_step=False,subset_generator=None):
    """
            Routine to reconstruct an image from a castor data file. This uses
            CUDA projectors and backprojectors.

            Positional Parameters:
                - cdh_name: full path to sinogram header (string) ; used if dict_sino is None.
                - cdf_path: directory where sinogram CDF is stored (string) ; used if dict_sino is None.
            Keyword Parameters:
                - reconsParams: parameters used for reconstruction (GPUreconsParams,
                    default:None ie default parameters for Biograph).
                - verbose: verbosity (boolean, default:True).
                - xstart: starting value for xstart (np array of same size as output, default:True).
                - dict_sino: dictionary containing sinograms (dictionary of tensors, default:None)
                  if None, this will be read by GPUCASToRBiographSinos using cdh_name, cdf_path.
                - eval_mode: if Fwd/Bwd modules used with autograd capability (boolean, default:True).
                - show_step: display the different steps for each iteration (boolean, default:False)
                - subset_generator: generator returning the subset number at each iteration (generator,
                    default:False is returning a cyclic subset generator).

            Returns:
                - full path to output image header (string).
                - stdout: standard output redirection (string).
                - stderr: standard error redirection (string).
    """

    #Initialize objects
    if reconsParams is None:
        reconsParams=GPUBiographReconsParams()
    if not isinstance(reconsParams,GPUBiographReconsParams):
        raise TypeError("BiographReconsParams type is not GPUBiographReconsParams ")
    BiographReconsOps=GPUReconsOps(reconsParams)
    if dict_sino is None:
        dict_sino=GPUCASToRSinos(cdh_name, cdf_path,reconsParams=reconsParams)
    im_tensor_size=reconsParams.tb_im_dim
    if verbose:
        print(f"TSize={im_tensor_size}")
        print(f"XYZ={reconsParams.XYZ}")
        print(f"VirtualRing={reconsParams._GPUReconsParams__VirtualRing}")
        print(f"VirtualRing={BiographReconsOps.Params._GPUReconsParams__VirtualRing}")

    y_data =  reconsParams.createSubsetData(dict_sino["val"])
    mult_array =  reconsParams.createSubsetData(dict_sino["mult"])
    add_array =  reconsParams.createSubsetData(dict_sino["add"])
    sinomask =  reconsParams.createSubsetData(dict_sino["mask"])
    deviceGPU=reconsParams.deviceGPU

    #Initialize image
    if xstart is None:
        x_em = torch.ones(im_tensor_size, dtype = torch.float32, device = deviceGPU)*reconsParams.start_val
    else:
        x_em= torch.reshape(torch.tensor(xstart,dtype=torch.float32,device=deviceGPU),im_tensor_size)
    if verbose:
        print(f"size x_em={x_em.size()}")

    if subset_generator is None:
        subset_generator=BiographReconsOps.getCyclicSubsetGenerator()

    sen=BiographReconsOps.computeSensitivityImages(sinomask,mult_array)
    sen = [s.to(deviceGPU) for s in sen]

    min_den=reconsParams.den_thr
    torch.cuda.synchronize(deviceGPU)
    tmax_up=torch.tensor(reconsParams.max_up, dtype = torch.float32, device = deviceGPU)
    tmin_up=torch.tensor(reconsParams.min_up, dtype = torch.float32, device = deviceGPU)

    if eval_mode:
        fwd_func=BiographReconsOps.forwardProj
        bwd_func=BiographReconsOps.backwardProj
    else:
        fwd_func=BiographReconsOps.forwardProj_autograd
        bwd_func=BiographReconsOps.backwardProj_autograd

    nbsubset=max([1,reconsParams.nsubsets])
    with tqdm(total=reconsParams.nit*nbsubset, desc="Iterations EM", unit='Iteration(s)') as pbar:
        for kit in range(reconsParams.nit*nbsubset):
            subset=next(subset_generator)
            if verbose:
                pbar.write(f"kitstart={kit//nbsubset}, subset={subset}, mem={torch.cuda.memory_allocated()/1e6}")
            fwd=fwd_func(x_em,mult_factor=mult_array[subset],add_factor=add_array[subset],subset=subset)
            if show_step:
                display3D(Tensor2Np(fwd),title=f"fwd {kit}")

            ratio=torch.where(fwd>min_den,torch.divide(y_data[subset],fwd),torch.tensor(1.0,
                                dtype = torch.float32, device = deviceGPU))*sinomask[subset]
            if show_step:
                display3D(Tensor2Np(ratio),title=f"ratio {kit}")
            torch.cuda.synchronize(deviceGPU)
            bwd=bwd_func(ratio,mult_factor=mult_array[subset],add_factor=0.,subset=subset)
            del fwd
            if show_step:
                display3D(Tensor2Np(bwd),title=f"bwd {kit}")
            bwd=torch.divide(bwd,sen[subset])
            if reconsParams.max_up >0:
                bwd=torch.where(bwd>tmax_up,tmax_up,bwd)
            if reconsParams.min_up >0:
                bwd=torch.where(bwd>tmin_up ,bwd,tmin_up)
            x_em=x_em*bwd
            if show_step:
                display3D(Tensor2Np(bwd),title=f"update {kit}")
            del bwd
            torch.cuda.synchronize(deviceGPU)
            torch.cuda.empty_cache()
            if show_step:
                display3D(Tensor2Np(x_em),title=f"x_em {kit}")
            #torch.cuda.synchronize(deviceGPU)
            if verbose:
                pbar.write(f"kitend={kit//nbsubset}, subset={subset}, mem={torch.cuda.memory_allocated()/1e6}")
            torch.cuda.synchronize(deviceGPU)
            pbar.update(1)

    x_em_cpu=np.swapaxes(np.squeeze(x_em.cpu().numpy()), 2, 0)
    del x_em,sen
    torch.cuda.empty_cache()

    return x_em_cpu



def GPUBiographProxRecons(BiographReconsParams=None,
                          xprox=None,
                          xstart=None, 
                          pnlt_beta=10,
                          dict_sino=None,
                          tensor_output=False,
                          img_mask = None,
                            ):


    #Initialize objects
    if not isinstance(BiographReconsParams,GPUBiographReconsParams):
        raise TypeError("BiographReconsParams type is not GPUBiographReconsParams ")
    BiographReconsOps=GPUReconsOps(BiographReconsParams)
    assert(dict_sino is not None)
    im_tensor_size=BiographReconsParams.tb_im_dim
    if pnlt_beta<=0:
        raise ValueError(f"pnlt_beta should be >0 instead of {pnlt_beta}")

    y_data =  BiographReconsParams.createSubsetData(dict_sino["val"])
    mult_array =  BiographReconsParams.createSubsetData(dict_sino["mult"])
    add_array =  BiographReconsParams.createSubsetData(dict_sino["add"])
    sinomask =  BiographReconsParams.createSubsetData(dict_sino["mask"])
    deviceGPU=BiographReconsParams.deviceGPU

    subset_generator=BiographReconsOps.getCyclicSubsetGenerator()

    #Initialize image
    if xstart is None:
        x_mapem = torch.ones(im_tensor_size, dtype = torch.float32, device = deviceGPU)*BiographReconsParams.start_val
    else:
        if not torch.is_tensor(xstart):
            x_mapem= torch.reshape(torch.tensor(np.ascontiguousarray(xstart),dtype=torch.float32,device=deviceGPU),im_tensor_size)
        else:
            x_mapem= torch.reshape(xstart.contiguous().cuda(),im_tensor_size)

    if img_mask is not None:
        img_mask = torch.tensor(np.swapaxes(img_mask,2,0),dtype=torch.float32,device=deviceGPU)

  
    if xprox is None:
        xprox = torch.zeros(im_tensor_size, dtype = torch.float32, device = deviceGPU)
    else:
        
        if not torch.is_tensor(xprox):
            xprox = np.swapaxes(xprox,2,0)
            xprox= torch.reshape(torch.tensor(np.ascontiguousarray(xprox),dtype=torch.float32,device=deviceGPU),im_tensor_size)
        else:
            xprox = torch.swapaxes(xprox,2,0)
            xprox= torch.reshape(xprox.contiguous().cuda(),im_tensor_size)
    
   
    #Compute sensitivity image
    sen=BiographReconsOps.computeSensitivityImages(sinomask,mult_factor=mult_array)  #No need for autograd here
    # if show_step:
    #     display3D(Tensor2Np(sen[0]),title="sen[0]")
    min_den=BiographReconsParams.den_thr
    torch.cuda.synchronize(deviceGPU)
    tmax_up=torch.tensor(BiographReconsParams.max_up, dtype = torch.float32, device = deviceGPU)
    tmin_up=torch.tensor(BiographReconsParams.min_up, dtype = torch.float32, device = deviceGPU)

    fwd_func=BiographReconsOps.forwardProj
    bwd_func=BiographReconsOps.backwardProj
   

    nbsubset=max([1,BiographReconsParams.nsubsets])
  
    with tqdm(total=BiographReconsParams.nit*nbsubset, desc="Iterations ProxRecons", unit='Iteration(s)') as pbar:
        for kit in range(BiographReconsParams.nit*nbsubset):
            subset=next(subset_generator)
            current_sen = sen[subset].to(deviceGPU)
           
            fwd=fwd_func(x_mapem,mult_factor=mult_array[subset],add_factor=add_array[subset],subset=subset)
            ratio=torch.where(fwd>min_den,torch.divide(y_data[subset],fwd),torch.tensor(1.0,
                                dtype = torch.float32, device = deviceGPU))*sinomask[subset]
            torch.cuda.synchronize(deviceGPU)
            bwd=bwd_func(ratio,mult_factor=mult_array[subset],add_factor=0,subset=subset)
            del fwd
            if BiographReconsParams.max_up >0:
                bwd=torch.where(bwd>tmax_up,tmax_up,bwd)
            if BiographReconsParams.min_up >0:
                bwd=torch.where(bwd>tmin_up ,bwd,tmin_up)
            
            if img_mask is not None:
                bwd = bwd * img_mask
                
            x_em_nos=x_mapem*bwd
            bwd=torch.divide(bwd,current_sen)
            x_em=x_mapem*bwd
            del bwd
            torch.cuda.synchronize(deviceGPU)
            torch.cuda.empty_cache()
            denom=((1.0-(pnlt_beta*xprox)/current_sen)+torch.sqrt(torch.pow((1.0-(pnlt_beta*xprox)/current_sen),2.0)+4*(pnlt_beta*x_em)/current_sen))
            x_mapem0=2.0*x_em/denom
            x_mapem1=((xprox-current_sen/pnlt_beta+torch.sqrt(torch.pow(xprox-current_sen/pnlt_beta,2.0)+4*x_em_nos/pnlt_beta))/2.0)
            x_mapem=torch.where(pnlt_beta*xprox<current_sen,x_mapem0,x_mapem1)
           
            torch.cuda.synchronize(deviceGPU)
            pbar.update(1)
            del x_em,x_em_nos
            if img_mask is not None :
                x_mapem = x_mapem * img_mask

    x_mapem = torch.swapaxes(x_mapem,2,4)
    if not tensor_output:
        x_mapem=np.squeeze(x_mapem.cpu().numpy())
        # img_mask = np.squeeze(img_mask.cpu().numpy())
    torch.cuda.empty_cache()
    del sen

    return x_mapem

