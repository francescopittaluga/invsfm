# Copyright (c) Microsoft Corporation.
# Copyright (c) University of Florida Research Foundation, Inc.
# Licensed under the MIT License.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in 
# the Software without restriction, including without limitation the rights to 
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
#
# tf_load_data.py
# Tensorflow functions for loading invsfm data
# Author: Francesco Pittaluga

import numpy as np
import tensorflow as tf


# Load Images & Depth Maps
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def num_digits(x):
    return tf.floor(log10(x)) + 1

def load_img(fp,dtype=None,binary=False):
    img = tf.read_file(fp)
    if not binary:
        code = tf.decode_raw(img,tf.uint8)[0]
        img = tf.cond(tf.equal(code,137),
                      lambda: tf.image.decode_png(img,channels=3),
                      lambda: tf.image.decode_jpeg(img,channels=3))
    else:
        hwc = tf.string_split([img],delimiter='&').values[:3]
        w = tf.string_to_number(hwc[0],out_type=tf.float32)
        h = tf.string_to_number(hwc[1],out_type=tf.float32)
        c = tf.string_to_number(hwc[2],out_type=tf.float32)
        start = tf.cast(3+num_digits(w)+num_digits(h)+num_digits(c),tf.int64)
        img = tf.substr(img,start,-1)
        img = tf.cast(tf.decode_raw(img,tf.float16),tf.float32)
        img = tf.reshape(img,tf.cast(tf.stack([h,w,c]),tf.int64))
    return img

# Load binary files
def load_bin_file(fp,dtype,shape):
    data = tf.read_file(fp)
    data = tf.decode_raw(data,dtype)
    data = tf.reshape(data,shape)
    return data

# Load camera from binary file
def load_camera(fp):
    cam = load_bin_file(fp,tf.float32,[23])
    K = tf.reshape(cam[:9],(3,3))
    R = tf.reshape(cam[9:18],(3,3))
    T = tf.reshape(cam[18:21],(3,1))
    h = cam[21]
    w = cam[22]
    return K,R,T,h,w

# set scale and crop for data augmentation
def scale_crop(h,w,crxy,crsz,scsz,isval,niter=0):
    scsz = tf.constant(np.float32(scsz),dtype=tf.float32)
    hw = tf.stack([h,w])
    if isval:
        sc = scsz[0]/tf.reduce_min(hw)
        new_sz = tf.to_int32(tf.ceil(sc*hw))
        cry = (new_sz[0]-crsz)//2
        crx = (new_sz[1]-crsz)//2
    else:
        sc = tf.random_shuffle(scsz,seed=niter)[0]/tf.reduce_min(hw)
        new_sz = tf.to_int32(tf.ceil(sc*hw))
        cry = tf.cast(tf.floor(crxy[0]*tf.to_float(new_sz[0]-crsz)),tf.int32)
        crx = tf.cast(tf.floor(crxy[1]*tf.to_float(new_sz[1]-crsz)),tf.int32)
    return sc,new_sz,cry,crx

# load and augment (random scale & crop) image batch 
def load_img_bch(img_paths,crsz,scsz,niter=0,isval=False,binary=False):
    img_batch = []
    crxy = tf.random_uniform([len(img_paths),2],minval=0.,maxval=1.,seed=niter)
    for i in range(len(img_paths)):
        img = load_img(img_paths[i],binary=binary)
        h = tf.to_float(tf.shape(img)[0])
        w = tf.to_float(tf.shape(img)[1])
        nch = tf.shape(img)[2]
        _,dep_sz,dep_cry,dep_crx = scale_crop(h,w,crxy[i],crsz,scsz,isval,niter)
        img = tf.image.resize_images(img,dep_sz)
        img = img[dep_cry:dep_cry+crsz,dep_crx:dep_crx+crsz,:]
        if not isval:
            img = tf.image.random_flip_left_right(img,seed=niter)
        img_batch.append(tf.reshape(img,[1,crsz,crsz,nch]))
    return tf.concat(img_batch,axis=0)

# load batch of sfm projections (xyz, color, depth, sift descriptor)
def load_proj_bch(camera_paths,pcl_xyz_paths,pcl_sift_paths,pcl_rgb_paths,
                  crsz,scsz,isval=False,niter=0):

    bsz = len(camera_paths)
    proj_depth_batch = []
    proj_sift_batch = []
    proj_rgb_batch = []

    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    crxy = tf.random_uniform([bsz,2],minval=0.,maxval=1.,seed=niter)

    for i in range(bsz):
        # load data from files
        K,R,T,w,h = load_camera(camera_paths[i])
        pcl_xyz = load_bin_file(pcl_xyz_paths[i],tf.float32,[-1,3])
        pcl_sift = tf.cast(load_bin_file(pcl_sift_paths[i],tf.uint8,[-1,128]),tf.float32)
        pcl_rgb = tf.cast(load_bin_file(pcl_rgb_paths[i],tf.uint8,[-1,3]),tf.float32)
        sc,_,cry,crx = scale_crop(h,w,crxy[i],crsz,scsz,isval,niter)

        # project pcl
        P = tf.matmul(K,tf.concat((R,T),axis=1))
        xyz_world = tf.concat((pcl_xyz,tf.ones([tf.shape(pcl_xyz)[0],1])),axis=1)
        xyz_proj = tf.transpose(tf.matmul(P,tf.transpose(xyz_world)))
        z = xyz_proj[:,2]
        x = xyz_proj[:,0]/z
        y = xyz_proj[:,1]/z

        mask_x = tf.logical_and(tf.greater(x,-1.),tf.less(x,tf.to_float(w)))
        mask_y = tf.logical_and(tf.greater(y,-1.),tf.less(y,tf.to_float(h)))
        mask_z = tf.logical_and(tf.greater(z,0.),tf.logical_not(tf.is_nan(z)))
        mask = tf.logical_and(mask_z,tf.logical_and(mask_x,mask_y))

        proj_x = tf.boolean_mask(x,mask)
        proj_y = tf.boolean_mask(y,mask)
        proj_z = tf.boolean_mask(z,mask)

        proj_depth = tf.expand_dims(proj_z,axis=1)
        proj_sift = tf.boolean_mask(pcl_sift,mask,axis=0)
        proj_rgb = tf.boolean_mask(pcl_rgb,mask,axis=0)

        # scale pcl
        proj_x = tf.round(proj_x*sc)
        proj_y = tf.round(proj_y*sc)
        h *= sc
        w *= sc

        #################
        # sort proj tensor by depth (descending order)
        _,inds_global_sort = tf.nn.top_k(-1.*proj_z,k=tf.shape(proj_z)[0])
        proj_x = tf.gather(proj_x,inds_global_sort)
        proj_y = tf.gather(proj_y,inds_global_sort)

        # per pixel depth buffer
        seg_ids = tf.cast(proj_x*tf.cast(w,tf.float32) + proj_y, tf.int32)
        data = tf.range(tf.shape(seg_ids)[0])
        inds_pix_sort = tf.unsorted_segment_min(data,seg_ids,tf.reduce_max(seg_ids))
        inds_pix_sort = tf.boolean_mask(inds_pix_sort,tf.less(inds_pix_sort,INT32_MAX))

        proj_depth = tf.gather(tf.gather(proj_depth,inds_global_sort),inds_pix_sort)
        proj_sift = tf.gather(tf.gather(proj_sift,inds_global_sort),inds_pix_sort)
        proj_rgb = tf.gather(tf.gather(proj_rgb,inds_global_sort),inds_pix_sort)

        h = tf.cast(h,tf.int32)
        w = tf.cast(w,tf.int32) 
        proj_yx = tf.cast(tf.concat((proj_y[:,None],proj_x[:,None]),axis=1),tf.int32)
        proj_yx = tf.gather(proj_yx,inds_pix_sort)

        proj_depth = tf.scatter_nd(proj_yx,proj_depth,[h,w,1])
        proj_sift = tf.scatter_nd(proj_yx,proj_sift,[h,w,128])
        proj_rgb = tf.scatter_nd(proj_yx,proj_rgb,[h,w,3])
        ################

        # crop proj
        proj_depth = proj_depth[cry:cry+crsz,crx:crx+crsz,:]
        proj_sift = proj_sift[cry:cry+crsz,crx:crx+crsz,:]
        proj_rgb = proj_rgb[cry:cry+crsz,crx:crx+crsz,:]

        # randomly flip proj
        if not isval:
            proj_depth = tf.image.random_flip_left_right(proj_depth,seed=niter)
            proj_sift = tf.image.random_flip_left_right(proj_sift,seed=niter)
            proj_rgb = tf.image.random_flip_left_right(proj_rgb,seed=niter)

        proj_depth_batch.append(proj_depth)
        proj_rgb_batch.append(proj_rgb)
        proj_sift_batch.append(proj_sift)
    
    return proj_depth_batch, proj_sift_batch, proj_rgb_batch
