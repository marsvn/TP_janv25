import os
import re
import errno

import numpy as np
import os,sys
from interfile import Interfile
import copy

"""
Created on October 2021

@author: florent.sureau
"""


def getNPArrayFromCASToRInterfileHeader(header,path,strict_enforce=True,verbose=True):
    """
            Routine to extract image from binary file using dictionary build using interfile package.
            Note: a lot of keywords are not implemented, this routine should read interfile for/produced by CASToR
            Positional Parameters:
                - header: dictionary built by interfile package from header (interfile dictionary)
                - path: path of the binary file (string)

            Returns:
                - image (numpy.ndarray)
    """

    ff=''
    CastorKeyList=['name of data file','originating system','PET data type',
             'data format','number format','number of bytes per pixel',
             'Patient name','number of dimensions','image duration',
             'image start time']
    CastorkeyMandatory=[True,False,False,
                   False,False,False,
                   False,True,False,
                   False]
    notHandledList=['data compression', 'data encode',
                   'NUD/rescale slope','NUD/rescale intercept']
    OtherKeys=['slice orientation','orientation','data starting block','data offset in bytes','imagedata byte order','PET data type']

    offset=0
    try:
        for kkey,key in enumerate(CastorKeyList):
            if key not in header.keys():
            #Ensure we have mandatory keys
                if CastorkeyMandatory[kkey]:
                    print(key,header[key]['value'])
                    raise KeyError("No mandatory key {0} ".format(key))
            else:
                if not CastorkeyMandatory[kkey]:
                    if verbose:
                        print("Optional Key '{0}': {1}".format(key,header[key]['value']))

        for kkey,key in enumerate(notHandledList):
            if key in header.keys():
                if verbose:
                    print('HND KEY')
                if (header[key]['value'] is not None):
                    print("WARNING: NOT HANDLED NHkey=",header[key]['value'])
                    if strict_enforce:
                        print('STRICT ENFORCE')
                        raise KeyError("Not handled key {0}:{1} ".format(key,header[key]['value']))

        #Get Filename
        filename=os.path.join(path,header['name of data file']['value'])
        if verbose:
            print('Open {0}'.format(filename))
        #Get Format
        if 'imagedata byte order' in header.keys():
            if(header['imagedata byte order']['value'].lower()=="LITTLEENDIAN".lower()):
                ff+='<'
            elif(header['imagedata byte order']['value'].lower()=="BIGENDIAN".lower()):
                ff+='>'
            else:
                raise TypeError("Unknow order {0}".format(header['imagedata byte order']['value']))
        else:
            if verbose:
                print("ASSUME LITTLE ENDIAN")
            ff+='<'
        if 'number format' in header.keys():
            if "float" in header['number format']['value'].lower():
                ff+='f'
            elif "unsigned integer" in header['number format']['value'].lower():
                ff+='u'
            elif "signed integer" in header['number format']['value'].lower():
                ff+='i'
            else :
                raise TypeError("Unknow type {0}".format(header['number format']['value']))
        else:
            ff+='f' #default value
        if 'number of bytes per pixel' in header.keys():
            ff+=str(header['number of bytes per pixel']['value'])
        else:
            ff+='4'#default value
        if verbose:
            print("string format {0}".format(ff))


        #Get Number of dimensions
        ndims=int(header['number of dimensions']['value'])
        lst_dims=[]
        scale_fact=[]
        bnd_or=[]
        fov_sz=[]
        if verbose:
            print("Ndims=",ndims)
        for kdim in range(1,ndims+1):
            if 'matrix size [{0}]'.format(kdim) not in header.keys():
                if 'matrix size[{0}]'.format(kdim) not in header.keys():
                    raise KeyError("Missing key 'matrix size [{0}]' ".format(kdim))
                else:
                    if verbose:
                        print("kdim={0}, mat={1}".format(kdim,header['matrix size[{0}]'.format(kdim)]['value']))
                    lst_dims.append(header['matrix size[{0}]'.format(kdim)]['value'])
            else:
                if verbose:
                    print("kdim={0}, mat={1}".format(kdim,header['matrix size [{0}]'.format(kdim)]['value']))
                lst_dims.append(header['matrix size [{0}]'.format(kdim)]['value'])
            if 'scaling factor (mm/pixel) [{0}]'.format(kdim) in header.keys():
                scale_fact.append(header['scaling factor (mm/pixel) [{0}]'.format(kdim)]['value'])
            if 'bounding origin (mm) [{0}]'.format(kdim) in header.keys():
                bnd_or.append(header['bounding origin (mm) [{0}]'.format(kdim)]['value'])
            if 'FOV size (mm) [{0}]'.format(kdim) in header.keys():
                fov_sz.append(header['FOV size (mm) [{0}]'.format(kdim)]['value'])
        lst_dims.reverse()
        if len(scale_fact)>0:
            scale_fact.reverse()
        if len(bnd_or)>0:
            bnd_or.reverse()
        if len(fov_sz)>0:
            fov_sz.reverse()
        if verbose:
            print("lst_dims=",lst_dims)
        ktotal=1
        for kd in lst_dims:
            ktotal*=kd

        if 'data starting block' in header.keys():
            offset=2048*header['data starting block']['value']
            if verbose:
                print("offset={0}".format(offset))
        elif 'data offset in bytes' in header.keys():
            offset=header['data offset in bytes']['value']
            if verbose:
                print("offset={0}".format(offset))

        #Read all binary file and reshape it
        with open(filename, 'rb') as f:
            f.seek(offset)
            data=np.fromfile(f,dtype=ff,count=ktotal)
        data=np.reshape(data,lst_dims,order='C')
        for kk,key in enumerate(header.keys()):
            if 'data format' in key.lower():
                if verbose:
                    print(header[key]['value'])
                # if('image' not in header[key]['value'].lower()):
                #     tmp=lst_dims[ndims-2]
                #     lst_dims[ndims-2]=lst_dims[ndims-1]
                #     lst_dims[ndims-1]=tmp
                #     data=np.swapaxes(data, ndims-2, ndims-1)
                #     if(len(scale_fact)==ndims):
                #         tmp=scale_fact[ndims-2]
                #         scale_fact[ndims-2]=scale_fact[ndims-1]
                #         scale_fact[ndims-1]=tmp
                #     if(len(bnd_or)==ndims):
                #         tmp=bnd_or[ndims-2]
                #         bnd_or[ndims-2]=bnd_or[ndims-1]
                #         bnd_or[ndims-1]=tmp
                #     if(len(fov_sz)==ndims):
                #         tmp=fov_sz[ndims-2]
                #         fov_sz[ndims-2]=fov_sz[ndims-1]
                #         fov_sz[ndims-1]=tmp
            if 'orientation' in key.lower():
                if verbose:
                    print("Optional Key '{0}': {1}".format(key,header[key]['value']))
                   #Transverse[default]|Coronal|Sagittal|
        if verbose:
            print("Ndims: {0}".format(ndims))
            print("Dimensions: {0}".format(lst_dims))
        return data,lst_dims,scale_fact,bnd_or,fov_sz

    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1


def writeInterfileImgFromHDR(out_dir,out_fname,in_dir,hdrname,img,verbose=True):
    """
            Routine to write an array as interfile IMG with a hdr
            Positional Parameters:
                - out_dir: directory where the img will be saved (string)
                - out_fname: name of file where interfile image is saved (string)
                - in_dir:  directory where the reference header is located (string)
                - hdrname: name of header used as refererence (string)
                - img: data to be saved (numpy array)
                - verbose: verbosity (boolean, default: True)
    """

    if out_fname.endswith('.ima'):
        new_hdrname=out_fname.replace(".ima",".hdr")
    elif out_fname.endswith('.img'):
        new_hdrname=out_fname.replace(".img",".hdr")
    elif out_fname.endswith('.i'):
        new_hdrname=out_fname.replace(".i",".hdr")
    elif out_fname.endswith('.s'):
        new_hdrname=out_fname.replace(".s",".s.hdr")
    else:
        print("extension not handled")
    intFlag=False
    uintFlag=False
    shortFlag=False
    if verbose:
        print("WRITE:",os.path.join(out_dir,new_hdrname)," READ:",os.path.join(in_dir,hdrname))
    orderLine=False
    bytes_per_pixel=0
    with open(os.path.join(in_dir,hdrname), "r") as f_in:
        with open(os.path.join(out_dir,new_hdrname), "w") as f_out:
            for line in f_in:
                stripped_line = line.strip()
                split_line = stripped_line.split(":=")
                if "!name of data file" in (split_line[0]).strip():
                    f_out.write("!name of data file := {0}\n".format(out_fname))
                    if verbose:
                        print("write=",out_fname)
                elif 'number format' in (split_line[0]).strip():
                    if verbose:
                        print(f" lined_strip={(split_line[1]).strip()}")
                    if ("unsigned integer") in (split_line[1]).strip():
                        uintFlag=True
                    elif ("signed integer") in (split_line[1]).strip():
                        intFlag=True
                    elif ("float") in (split_line[1]).strip():
                        intFlag=False
                        if ("short") in (split_line[1]).strip():
                            shortFlag=True
                    else:
                        raise TypeError("Unknow type {0}".format((split_line[1]).strip()))
                    f_out.write(line)
                else:
                    if 'imagedata byte order' in (split_line[0]).strip():
                        orderLine=True
                        if sys.byteorder=="little":
                            f_out.write("!imagedata byte order :=LITTLEENDIAN\n")
                        else:
                            f_out.write("!imagedata byte order :=BIGENDIAN\n")
                    elif 'matrix size [1]' in (split_line[0]).strip():
                        f_out.write("matrix size [1] :={0}\n".format(img.shape[2]))
                    elif 'matrix size [2]' in (split_line[0]).strip():
                        f_out.write("matrix size [2] :={0}\n".format(img.shape[1]))
                    elif 'matrix size [3]' in (split_line[0]).strip():
                        f_out.write("matrix size [3] :={0}\n".format(img.shape[0]))
                    elif 'bytes per pixel' in (split_line[0]).strip():
                        bytes_per_pixel=int((split_line[1]).strip())
                        if verbose:
                            print("bytes_per_pixel",bytes_per_pixel)
                        f_out.write(line)
                    else:
                        f_out.write(line)
            if orderLine==False:
                if sys.byteorder=="little":
                    f_out.write("!imagedata byte order := LITTLEENDIAN\n")
                else:
                    f_out.write("!imagedata byte order := BIGENDIAN\n")

    if (bytes_per_pixel>0):
        if intFlag or uintFlag:
            if bytes_per_pixel==2:
                shortFlag=True
            elif bytes_per_pixel==4:
                shortFlag=False
        else:
            if bytes_per_pixel==4:
                shortFlag=True
            elif bytes_per_pixel==8:
                shortFlag=False
    if verbose:
        print(f"SHORTFLAG={shortFlag},uintFlag={uintFlag},intFlag={intFlag}")
    if intFlag:
        if shortFlag:
            img_cvt=img.astype("int16")
        else:
            img_cvt=img.astype("int32")
        img_cvt.tofile(os.path.join(out_dir,out_fname))
    elif uintFlag:
        if shortFlag:
            img_cvt=img.astype("uint16")
        else:
            img_cvt=img.astype("uint32")
        img_cvt.tofile(os.path.join(out_dir,out_fname))
    else:
        if shortFlag:
            img_cvt=img.astype("float32")
        else:
            img_cvt=img.astype("float64")
        img_cvt.tofile(os.path.join(out_dir,out_fname))

def writeInterfileImg(array,filepath,pixsize=[2.027,2.03642,2.03642],
    scanner="PET_SIEMENS_BIOGRAPH6_TRUEPOINT_TRUEV",patient_name="0",
    patient_id="0",save_type=None,flip=False):
    """
            Routine to write an array as interfile IMG with a minimal size hdr
            Positional Parameters:
                - array: data to be saved
                - filepath: filepath of file where interfile image will be saved (string)
            Keyword Parameters:
                - pixsize:  pixel sizes (list of floats, default: [2.027,2.03642,2.03642], z first)
                - scanner: name of PET scanner used (string)
                - patient_name: name of patient (string)
                - patient_id: id of patient (string)
                - save_type: type of data to be saved (string, numpy dtype - if None: "<f4")
    """
    try:
        ref_fname=os.path.basename(u"{0}".format(filepath))
        hdr_name=os.path.splitext(filepath)[0]+'.hdr'
        #hdr_name=filename+'.hdr'
        data=array
        ndim=len(np.shape(data))
        dims=np.shape(data)
        pixdim=pixsize
        if ndim>4:
            raise TypeError("Input dimensions {0} should be <=4".format(ndim))
        elif ndim>3:
            nims=np.shape(data)[3]
            ndim=3
        else:
            nims=1
        for k in range(ndim):
            if dims[k] !=  np.shape(data)[k]:
                raise TypeError("Input dim {0} should be {1}".format(dims[k],np.shape(data)[k]))

        #Copy array to file
        if save_type is None:
            save_type="<f4"
        dt = np.dtype(save_type)
        if not (dt.str=="<f4" or dt.str=="<f8" or dt.str=="<i2" or dt.str=="<u2"):
            raise TypeError("Format {0} not supported (only <f4 or <f8 or <i2 or <u2 supported)".format(dt.str))

        if(flip):
            #This saves in RAS convention
            cpdata=copy.deepcopy(np.flip(data.astype(save_type, order='C').T))
            cpdata.tofile(filepath, sep="")
        else:
            dims=copy.deepcopy(list(np.shape(data)))
            dims.reverse()
            pixdim=copy.deepcopy(pixsize)
            pixdim.reverse()
            cpdata=copy.deepcopy(data.astype(save_type, order='C'))
            cpdata.tofile(filepath, sep="")

        #Create Header
        header_lines=[]
        header_lines.append("!INTERFILE :=" + "\n")
        header_lines.append("!version of keys :=3.3" + "\n")
        header_lines.append("!GENERAL IMAGE DATA :=" + "\n")
        header_lines.append("!name of data file :={0}".format(ref_fname)+'\n')
        header_lines.append("!PET data type :=emission"+'\n')
        header_lines.append("data format :=image"+'\n')
        header_lines.append("!originating system :={0}".format(scanner)+'\n')
        header_lines.append("!patient ID :={0}".format(patient_id)+'\n')
        header_lines.append("!patient name :={0}".format(patient_name)+'\n')
        header_lines.append("slice orientation :=Transverse"+'\n')
        header_lines.append("!imagedata byte order :=LITTLEENDIAN"+'\n')
        header_lines.append("data compression :=none"+'\n')
        header_lines.append("data encode :=none"+'\n')
        if dt.str=="<f4":
            header_lines.append("!number format :=short float"+'\n')
            header_lines.append("!number of bytes per pixel :=4"+'\n')
        elif dt.str=="<f8":
            header_lines.append("!number format :=long float"+'\n')
            header_lines.append("!number of bytes per pixel :=8"+'\n')
        elif dt.str=="<i2":
            header_lines.append("!number format :=signed integer"+'\n')
            header_lines.append("!number of bytes per pixel :=2"+'\n')
        elif dt.str=="<u2":
            header_lines.append("!number format :=unsigned integer"+'\n')
            header_lines.append("!number of bytes per pixel :=2"+'\n')
        else:
            raise TypeError("Format {0} not supported".format(dims[k],np.shape(data)[k]))
        header_lines.append("!total number of images :={0}".format(nims)+'\n')
        header_lines.append("number of dimensions :={0}".format(ndim)+'\n')
        for k in range(ndim):
            header_lines.append("matrix size [{0}] :={1}".format(k+1,dims[k])+'\n')
        for k in range(ndim):
            header_lines.append("scaling factor (mm/pixel) [{0}] :={1}".format(k+1,pixdim[k])+'\n')
        header_lines.append("image data unit := "+'\n')
        header_lines.append("!END OF INTERFILE :="+'\n')
        with open(hdr_name, "w",encoding="ascii") as fh:
            fh.writelines(header_lines)
        return 0

    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1

def writeInterfileSino(data,filepath,scanner="Biograph",patient_name="0",patient_id="0",
    scale_factor=None,save_type=None,hdr_ref=None,calib_factor=None,maxRing=38,Span=11,Trues=True):
    """
            Routine to write an array as interfile sinogram with a minimal size hdr
            Positional Parameters:
                - data: data to be saved
                - filepath: filepath of file where interfile image will be saved (string)
            Keyword parameters:
                - scanner: name of PET scanner used (string)
                - patient_name: name of patient (string)
                - patient_id: id of patient (string)
                - scale_factor : bin size, if hdr_ref is not present (list of float)
                - save_type: type of data to be saved (string, numpy dtype - if None: "<f4")
                - hdr_ref : header to be used as reference (interfile Header)
                - calib_factor: calibration factor is used to rescale the sinogram (float)
    """
    try:
        ref_fname=os.path.basename(u"{0}".format(filepath))
        hdr_name=os.path.splitext(filepath)[0]+'.hdr'
        #hdr_name=filepath+'.hdr'
        ndim=len(np.shape(data))
        dims=copy.deepcopy(list(np.shape(data)))
        dims.reverse()
        if scale_factor is None:
            if hdr_ref is not None:
                pixdim=[]
                ref_keys=hdr_ref.keys()
                lst_keys_ref=['scaling factor (mm/pixel) [1]','scaling factor (mm/pixel) [2]',
                              'scaling factor (mm/pixel) [3]']
                for kk,kkey in enumerate(lst_keys_ref):
                    if kkey in ref_keys:
                        pixdim.append(float(hdr_ref[kkey]["value"]))
            else:
                pixdim=[1,1,1]
        else:
            pixdim=copy.deepcopy(scale_factor)
            pixdim.reverse()
        #print("scale_factor=",scale_factor," pixdim=",pixdim)
        nims=1
        ndim=3

        if save_type is None:
            save_type="<f4"
        dt = np.dtype(save_type)
        if not (dt.str=="<f4" or dt.str=="<f8" or dt.str=="<i2" or dt.str=="<u2"):
            raise TypeError("Format {0} not supported (only <f4 or <f8 or <i2 or <u2 supported)".format(dt.str))

        #This saves in RAS convention
        #cpdata=copy.deepcopy(np.flip(data.astype(save_type, order='C').T))
        cpdata=copy.deepcopy(data.astype(save_type, order='C'))
        cpdata.tofile(filepath, sep="")

        #Create Header
        header_lines=[]
        header_lines.append("!INTERFILE :=" + "\n")
        header_lines.append("!version of keys :=3.3" + "\n")
        header_lines.append("!GENERAL IMAGE DATA :=" + "\n")
        header_lines.append("!name of data file :={0}".format(ref_fname)+'\n')
        header_lines.append("!PET data type :=emission"+'\n')
        header_lines.append("data format :=sinogram"+'\n')
        header_lines.append("!originating system :={0}".format(scanner)+'\n')
        header_lines.append("!Scanner name :={0}".format(scanner)+'\n')
        header_lines.append("!patient ID :={0}".format(patient_id)+'\n')
        header_lines.append("!patient name :={0}".format(patient_name)+'\n')
        header_lines.append("slice orientation :=Transverse"+'\n')
        header_lines.append("!imagedata byte order :=LITTLEENDIAN"+'\n')
        header_lines.append("data compression :=none"+'\n')
        header_lines.append("data encode :=none"+'\n')
        if hdr_ref is None:
            header_lines.append("axial compression:={0}".format(Span)+'\n')
            header_lines.append("maximum ring difference :={0}".format(maxRing)+'\n')
        if dt.str=="<f4":
            header_lines.append("!number format :=short float"+'\n')
            header_lines.append("!number of bytes per pixel :=4"+'\n')
        elif dt.str=="<f8":
            header_lines.append("!number format :=long float"+'\n')
            header_lines.append("!number of bytes per pixel :=8"+'\n')
        elif dt.str=="<i2":
            header_lines.append("!number format :=signed integer"+'\n')
            header_lines.append("!number of bytes per pixel :=2"+'\n')
        elif dt.str=="<u2":
            header_lines.append("!number format :=unsigned integer"+'\n')
            header_lines.append("!number of bytes per pixel :=2"+'\n')
        else:
            raise TypeError("Format {0} not supported".format(dims[k],np.shape(data)[k]))
        header_lines.append("!total number of images :={0}".format(nims)+'\n')
        header_lines.append("number of dimensions :={0}".format(ndim)+'\n')
        for k in range(ndim):
            header_lines.append("matrix size [{0}] :={1}".format(k+1,dims[k])+'\n')
        for k in range(ndim):
            header_lines.append("scaling factor (mm/pixel) [{0}] :={1:.5f}".format(k+1,pixdim[k])+'\n')

        if hdr_ref is not None:
            ref_keys=hdr_ref.keys()
            lst_keys_ref=['axial compression','maximum ring difference', 'mashing power', 'image duration',
             'image start time', 'Dose type', 'isotope halflife', 'branching factor','frame', 'bed',
             'Total Prompts', 'Total Randoms', 'Total Net Trues', 'Total Scatters', 'decay correction factor',
             'decay correction factor2', 'Dead time correction factor', 'start horizontal bed position']
            for kkey in lst_keys_ref:
                if kkey in ref_keys:
                    if kkey in['Dose type']:
                        if(isinstance(hdr_ref[kkey]["value"],str)):
                            header_lines.append("{0}:= {1}".format(kkey, hdr_ref[kkey]["value"])+'\n')
                        else:
                            header_lines.append("{0}:= {1}".format(kkey,str(hdr_ref[kkey]["value"]).upper())+'\n')
                    else:
                        header_lines.append("{0}:= {1}".format(kkey, hdr_ref[kkey]["value"])+'\n')
        header_lines.append("image data unit := "+'\n')
        if calib_factor is not None:
            header_lines.append("calibration factor := {0}".format(calib_factor)+'\n')
            header_lines.append("calibration factor for randoms := {0}".format(calib_factor)+'\n')
            header_lines.append("calibration unit := Bq/cc"+'\n')

        header_lines.append("!END OF INTERFILE :="+'\n')
        with open(hdr_name, "w",encoding="ascii") as fh:
            fh.writelines(header_lines)
        return 0

    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1


def readCastorHistoHeader(hdr_filename,fname_dir=None, verbose = False):
    """ This read the CASToR header of a histogram and extract relevant
    information as lists for later building the sinogram

    :param hdr_filename: filepath to header file
    :param fname_dir: dirpath where the castor data file is located (default: None, current directory)
    :type hdr_filename: string
    :type fname_dir: string

    :returns: dictionary of parameters in header
    :rtype: dictionary
    """

    ff=''
    CastorKeyList=['Data filename','Scanner name','Number of events',
             'Data mode','Data type','Start time (s)','Duration (s)',
             'Maximum number of lines per event','Axial compression',
             'Azymutal compression','Max ring diff','Calibration factor',
             'Isotope','Attenuation correction flag','Normalization correction flag',
             'Scatter correction flag','Random correction flag','TOF information flag']
    CastorkeyMandatory=[True]*7+[False]*11
    CastorkeyMissing=[True]*18
    #Default Values
    dic_init={}
    for key,kmand in zip(CastorKeyList,CastorkeyMandatory):
        if not kmand:
            dic_init[key]=list([''])
    dic_init['Isotope']='INF'
    dic_init['Calibration factor']='1'
    dic_init['Max ring diff']='38'
    dic_init['Azymutal compression']='1'
    dic_init['Axial compression']='11'
    dic_init['Maximum number of lines per event']='1'
    dic_init['Attenuation correction flag']='0'
    dic_init['Normalization correction flag']='0'
    dic_init['Scatter correction flag']='0'
    dic_init['Random correction flag']='0'
    dic_init['TOF information flag']='0'
    try:
        with open(hdr_filename) as f:
            lines=f.readlines()
        dic={}
        for line in lines:
            key,value,_=re.split("[:,\n]", line)
            if key in CastorKeyList:
                kindex=CastorKeyList.index(key)
                CastorkeyMissing[kindex]=False
                dic[key]=value.lstrip(' ')
            else:
                raise TypeError("Key {0} not recognized\n".format(key))
        for kn,kflag in enumerate(zip(CastorkeyMissing,CastorkeyMandatory)):
            ckey=CastorKeyList[kn]
            if(kflag[0] and kflag[1]):
                raise KeyError("Missing mandatory key {0}".format(ckey))
            else:
                if(kflag[0]):
                    if verbose:
                        print("Default initialize {0}".format(ckey))
                    dic[ckey]= dic_init[ckey]

        if(not 'histogram' in dic['Data mode']):
            raise KeyError("Mode should be histogram and not {0}".format(dic['Data mode']))
        if(dic['TOF information flag']=='1'):
            raise KeyError("No TOF available yet ({0})".format(dic['TOF information flag']))
        if fname_dir is not None:
            dic['Data filename']=os.path.join(fname_dir,dic['Data filename'])
        return dic
    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1

def cyReadCastorHisto(hdr_filename,fname_dir=None,ZeroFill=True,Endian="=",count=-1,
    verbose=False):
    """ This read the CASToR header of a histogram and extract the sinogram

    :param hdr_filename: filepath to header file
    :param fname_dir: dirpath where the castor data file is located
    :param ZeroFill: flag specifying whether ZeroFilling crystal pairs
                    (so that each bin has same number of crystal pairs)
    :type hdr_filename: string
    :type fname_dir: string
    :type ZeroFill: bool

    :returns: sinogram
    :rtype: 3D float np.array (2)
    .. warning:: only bins >0 are extracted
    """

    ff=''

    try:
        dic=readCastorHistoHeader(hdr_filename,fname_dir=fname_dir)
        #Create numpy dtype for reading
        fields_npy=[]
        nlines_index=None
        kindex=0
        fields_npy.append(('time', Endian+'u4'))
        kindex+=1
        if(dic['Attenuation correction flag']=='1'):
            fields_npy.append(('att', Endian+'f4'))
            kindex+=1
        if(dic['Random correction flag']=='1'):
            fields_npy.append(('rand', Endian+'f4'))
            kindex+=1
        if(dic['Normalization correction flag']=='1'):
            fields_npy.append(('norm', Endian+'f4'))
            kindex+=1
        fields_npy.append(('val', Endian+'f4'))
        val_index=kindex
        kindex+=1
        if(dic['Scatter correction flag']=='1'):
            fields_npy.append(('scat', Endian+'f4'))
            kindex+=1
        klines=int(dic['Maximum number of lines per event'])
        nbins=int(dic['Number of events'])
        if(klines>1):
            fields_npy.append(('nlines', Endian+'u2'))
            nlines_index=kindex
            kindex+=1
        if(ZeroFill):
            for kc in range(2*klines):
                fields_npy.extend([('crys{0}'.format(kc), Endian+'u4')])
                kindex+=1
        else:
            crys1_ind=kindex
            crys2_ind=crys1_ind+1
            for kc in range(2):
                fields_npy.extend([('crys{0}'.format(kc), Endian+'u4')])
                kindex+=1
        size_read=np.sum([np.dtype(kr).itemsize for kx,kr in fields_npy])
        if verbose:
            print("SZ_read=",size_read)
            print("fields_npy=",fields_npy)
        #Read directly data
        filepath=dic['Data filename']
        if verbose:
            print(filepath)
        if(ZeroFill):#Case zero filling
            #or(nlines_index is None) #or 1 single LOR per bin
            sino=cy_read_structured_file(filepath,fields_npy,dic['Max ring diff'],
                                         dic['Axial compression'],count,verbose)
        else:
            sino=cy_read_structured_file_line(filepath,fields_npy,dic['Max ring diff'],
                        dic['Axial compression'],count,val_index,crys1_ind,crys2_ind,
                        nlines_index,nbins,Endian,verbose)
        return sino
    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1

def cyReadCastorAllHisto(hdr_filename,fname_dir=None,ZeroFill=True,Endian="=",count=-1,
    multAdd=False,verbose=False):
    """ This read the CASToR header of a histogram and extract sinograms, including data
    and correction factors (ncf, acf, randoms rate, scatter rate, nLORs per bin).
    If multAdd is set, these sinograms are combined to get a multiplicative term (1/(ncf*acf),
    with a missing calibration/decay correction factor to be included later and also potentially
    divided by the nLORs if all LORs per projection bin included during projection;
     and an additive term  (randoms rate+scatter_rate, that ultimately needs to be multiplied by
    frame duration).

    :param hdr_filename: filepath to header file
    :param fname_dir: dirpath where the castor data file is located
    :param ZeroFill: flag specifying whether ZeroFilling crystal pairs
                    (so that each bin has same number of crystal pairs)
    :param count: extract the first count bins (-1 for all)
                    (so that each bin has same number of crystal pairs)
    :param multAdd: gather sinograms by multiplicative/additive factors
    :param verbose: verbosity
    :type hdr_filename: string
    :type fname_dir: string
    :type ZeroFill: bool
    :type count: int
    :type multAdd: bool
    :type verbose: bool

    :returns: dictionary of header info, dictionary of sinograms
    with potential keys "norm" (ncf), "att" (acf), "scat","rand", "nlines" or
    "mult" and "add"
    :rtype: dictionary of params, dictionary of 3D float np.array
    .. warning:: all bins are extracted
    """

    ff=''

    try:
        dic_header=readCastorHistoHeader(hdr_filename,fname_dir=fname_dir)
        #Create numpy dtype for reading
        fields_npy=[]
        dico_indices={}
        dico_indices["nlines_index"]=None
        dico_indices["val_index"]=None
        dico_indices["att_index"]=None
        dico_indices["norm_index"]=None
        dico_indices["scat_index"]=None
        dico_indices["rand_index"]=None
        dico_indices["crys0"]=None
        dico_indices["crys1"]=None
        kindex=0
        fields_npy.append(('time', Endian+'u4'))
        kindex+=1
        if(dic_header['Attenuation correction flag']=='1'):
            fields_npy.append(('att', Endian+'f4'))
            dico_indices["att_index"]=kindex
            kindex+=1
        if(dic_header['Random correction flag']=='1'):
            fields_npy.append(('rand', Endian+'f4'))
            dico_indices["rand_index"]=kindex
            kindex+=1
        if(dic_header['Normalization correction flag']=='1'):
            fields_npy.append(('norm', Endian+'f4'))
            dico_indices["norm_index"]=kindex
            kindex+=1
        fields_npy.append(('val', Endian+'f4'))
        dico_indices["val_index"]=kindex
        kindex+=1
        if(dic_header['Scatter correction flag']=='1'):
            fields_npy.append(('scat', Endian+'f4'))
            dico_indices["scat_index"]=kindex
            kindex+=1
        klines=int(dic_header['Maximum number of lines per event'])
        nbins=int(dic_header['Number of events'])
        if(klines>1):
            fields_npy.append(('nlines', Endian+'u2'))
            dico_indices["nlines_index"]=kindex
            kindex+=1
        dico_indices["crys0_index"]=kindex
        dico_indices["crys1_index"]=kindex+1
        if(ZeroFill):
            for kc in range(2*klines):
                fields_npy.extend([('crys{0}'.format(kc), Endian+'u4')])
                kindex+=1
        else:
            crys1_ind=kindex
            crys2_ind=crys1_ind+1
            for kc in range(2):
                fields_npy.extend([('crys{0}'.format(kc), Endian+'u4')])
                kindex+=1
        size_read=np.sum([np.dtype(kr).itemsize for kx,kr in fields_npy])
        if verbose:
            print("SZ_read=",size_read)
            print("fields_npy=",fields_npy)
        #Read directly data
        filepath=dic_header['Data filename']
        if verbose:
            print(filepath)
        if(ZeroFill):#Case zero filling
            #or(nlines_index is None) #or 1 single LOR per bin
            sino=cy_read_structured_file_allsino(filepath,fields_npy,dic_header['Max ring diff'],
                                         dic_header['Axial compression'],count,multadd=multAdd,verbose=verbose)
            return dic_header,sino
        else:
            #sino=cy_read_structured_file_line_allsino(filepath,fields_npy,dic_header['Max ring diff'],
            #            dic_header['Axial compression'],count,dico_indices,nbins,Endian,multAdd=multAdd,verbose=verbose)
            raise Exception("Not implemented as not supported anymore in CASToR")
    except Exception as inst:
        print(inst)
        print("EXCEPTION RAISED")
        return -1

def get_CASToR_config_radioisotope(name_radio,config_fpath=None):
    '''
        Get half life and branching ratio for a radio-isotope from CASToR config file.

        Arguments:
            - name_radio: name of radio-isotope (string)
            - config_fpath: filepath of radio-isotope constants (string, default: '${CASTOR_CONFIG}/misc/isotopes_pet.txt').
        Returns:
            - half-life of radio-isotope (s)
            - branching ratio of radio-isotope
    '''

    dic_radioisotopes={}
    if config_fpath is None:
        config_fpath=os.path.join(f"{os.getenv('CASTOR_CONFIG')}","misc/isotopes_pet.txt")
    if os.path.exists(config_fpath):
        with open(config_fpath, "rb") as f:
            txt=f.readlines()
            for kl in range(len(txt)):
                line=(txt[kl].decode()).split()
                if not line[0].startswith('#'):
                    ln_decode=[kx for kx in line if kx!='']
                    dic_radioisotopes[ln_decode[0]]={}
                    dic_radioisotopes[ln_decode[0]][name_cols[1]]=ln_decode[1]
                    dic_radioisotopes[ln_decode[0]][name_cols[2]]=ln_decode[2]
                elif kl==0:#name of columns
                    line=(txt[kl].decode()).split('\t')
                    name_cols=[kx.rstrip('\n') for kx in line if kx!='' and kx !='#']
        for kk in dic_radioisotopes.keys():
            if name_radio.lower()==kk.lower():
                for kkk in dic_radioisotopes[kk].keys():
                    if 'half-life' in kkk.lower():
                        HalfLife=dic_radioisotopes[kk][kkk]
                    if 'branching' in kkk.lower():
                        branchingRatio=dic_radioisotopes[kk][kkk]
        return float(HalfLife),float(branchingRatio)
    else:
        raise FileNotFoundError(errno.ENOENT,os.strerror(errno.ENOENT),config_fpath)


def compute_radioisotope_calibration_factor(frame_start,frame_duration,half_life=6586.2,branching_ratio=0.967,unit=1):
    '''
        Compute calibration factor taking into account decay correction factors
        and branching ratio (to be multiplied with data for decay/branching ratio correction).

        Arguments:
            - frame_start: start time for decay correction (float, in same unit as half_life -s-)
            - frame_duration: duration of frame (float, in same unit as half_life -s-)
            - half_life: half-life of radio-isotope (float, default: 6586.2 for F18)
            - branching_ratio: branching ratio of radio-isotope (float, default: 0.967 for F18)
            - unit: scale of activity concentration (float, default: 1 for Bq, should be 1000 for kBq).
        Returns:
            - multiplicative factor for decay correction and branching ratio correction
    '''
    lambda_iso=np.log(np.float32(2.0))/np.float32(half_life)
    decay_factor2=lambda_iso*np.float64(frame_duration)*np.exp(lambda_iso*np.float64(frame_start))/(1.0
        -np.exp(-lambda_iso*np.float64(frame_duration)))
    calfact = decay_factor2/(np.float64(frame_duration)*branching_ratio*unit)

    return calfact

def CASToRLoadAddMultFactors(castor_df_hdr,castor_df_dir,verbose=True):
    '''
    Load additive/multiplicative correction sinograms from a CASToR data file.
    This includes attenuation, normalization (assuming a projector per LOR and
    not per bin), calibration factor (ECF, decay) for multiplicative, and scatter
    and randoms for additive factor.

    Arguments:
        - castor_df_hdr:filetpath to CASToR datafile header (string).
        - castor_df_dir: path where CASToR datafile is (string).

    Returns:
        dictionary with keys "add" and "mult" containing multiplicative (1/(acf*ncf*calibration)) and
        additive factors ((scat+rand rates)*frame_duratoin) for projection.
        The model is y_i=mult_i  \sum_j h_{ij} x_j + add_i
        where y_i is the data in bin im, h_{ij} the projector weights for voxel j,
        x_j the activity concentation, and "add_i" and "mult_i" are given by dictionary.
    '''

    if castor_df_hdr.endswith('.cdf'):
        hdr_filepath=castor_df_hdr.replace(".cdf",".cdh")
    elif castor_df_hdr.endswith('.Cdf'):
        hdr_filepath=castor_df_hdr.replace(".Cdf",".Cdh")
    else:
        hdr_filepath=castor_df_hdr

    dict_hdr,dict_sino=cyReadCastorAllHisto(hdr_filepath,fname_dir=castor_df_dir,
        ZeroFill=True,Endian="=",count=-1,multAdd=True,verbose=False)

    half_life,branching_ratio=get_CASToR_config_radioisotope(dict_hdr['Isotope'])
    frame_duration=float(dict_hdr['Duration (s)'])
    frame_start=float(dict_hdr['Start time (s)'])
    calfact=compute_radioisotope_calibration_factor(frame_start,frame_duration,
        half_life=half_life,branching_ratio=branching_ratio,unit=1)
    if verbose:
        print(f"Frame Calfact={calfact}, ECF {dict_hdr['Calibration factor']}")
    calfact*=float(dict_hdr['Calibration factor'])


    dict_sino["add"]*=frame_duration
    dict_sino["mult"]/=calfact
    #CASToR homemade Forward proj includes all lines but not the normalization
    _=np.divide(dict_sino["mult"],dict_sino["nlines"],where=(dict_sino['nlines']>0),out=dict_sino['mult'])

    return dict_sino


def CASToR_load_sinograms(subject,real, doserec_object,verbose=True):
    '''
    Load additive/multiplicative correction sinograms from a CASToR data file.
    This includes attenuation, normalization (assuming a projector per LOR and
    not per bin), calibration factor (ECF, decay) for multiplicative, and scatter
    and randoms for additive factor.

    Arguments:
        - subject: subject name (string).
        - real: realization number (int, default: 0).
        - doserec_object: database descriptor (genSubDatabaseDoserecParams).
        - verbose: verbosity (bool, default: True)
    '''

    castor_df_ima,castor_df_hdr=doserec_object.get_fname_sinogram_CASToR_hdr(subject,real)
    castor_df_dir=doserec_object.get_gen_castor_dir(subject,real)

    castor_df_hdr_fpath=os.path.join(castor_df_dir,castor_df_hdr)

    return CASToRLoadAddMultFactors(castor_df_hdr_fpath,castor_df_dir,verbose=verbose)
