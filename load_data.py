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
# load_data.py
# Non-Tensorflow functions for loading invsfm data
# Author: Francesco Pittaluga

import numpy as np
import colmap.database as database
import colmap.read_model as read_model
from skimage.transform import resize
from skimage.io import imread

################################################################################
# Load sfm model directly from colmap output files 
################################################################################

# Load point cloud with per-point sift descriptors and rgb features from
# colmap database and points3D.bin file from colmap sparse reconstruction
def load_points_colmap(database_fp,points3D_fp):
    
    db = database.COLMAPDatabase.connect(database_fp)
    descriptors = dict(
        (image_id, database.blob_to_array(data, np.uint8, (rows, cols)))
        for image_id,data,rows,cols in db.execute("SELECT image_id, data, rows, cols FROM descriptors"))
    points3D = read_model.read_points3d_binary(points3D_fp)    
    keys = list(points3D.keys())
    
    pcl_xyz = []
    pcl_rgb = []
    pcl_sift = []
    for pt3D in points3D.values():
        pcl_xyz.append(pt3D.xyz)
        pcl_rgb.append(pt3D.rgb)
        i = np.random.randint(len(pt3D.image_ids),size=1)[0]
        pcl_sift.append(descriptors[pt3D.image_ids[i]][pt3D.point2D_idxs[i]])
        
    pcl_xyz = np.vstack(pcl_xyz).astype(np.float32)
    pcl_rgb = np.vstack(pcl_rgb).astype(np.uint8)
    pcl_sift = np.vstack(pcl_sift).astype(np.uint8)

    return pcl_xyz, pcl_rgb, pcl_sift


# Load camera matrices and names of corresponding src images from
# colmap images.bin and cameras.bin files from colmap sparse reconstruction
def load_cameras_colmap(images_fp,cameras_fp):

    images = read_model.read_images_binary(images_fp)
    cameras = read_model.read_cameras_binary(cameras_fp)
        
    src_img_nms=[]
    K = []; T = []; R = []; w = []; h = []
    
    for i in images.keys():
        R.append(read_model.qvec2rotmat(images[i].qvec))
        T.append((images[i].tvec)[...,None])
        k = np.eye(3)
        k[0,0] = cameras[images[i].camera_id].params[0]
        k[1,1] = cameras[images[i].camera_id].params[0]
        k[0,2] = cameras[images[i].camera_id].params[1]
        k[1,2] = cameras[images[i].camera_id].params[2]
        K.append(k)
        w.append(cameras[images[i].camera_id].width)
        h.append(cameras[images[i].camera_id].height)
        src_img_nms.append(images[i].name)
        
    return K,R,T,h,w,src_img_nms

################################################################################
# Load sfm model (and other data) from custom invsfm files
################################################################################

def load_camera(file_path):
    cam = np.fromfile(file_path,dtype=np.float32)
    K = np.reshape(cam[:9],(3,3))
    R = np.reshape(cam[9:18],(3,3))
    T = np.reshape(cam[18:21],(3,1))
    h = cam[21]
    w = cam[22]
    return K,R,T,h,w

def load_points_rgb(file_path):
    return np.reshape(np.fromfile(file_path, dtype=np.uint8),(-1,3))

def load_points_xyz(file_path):
    return np.reshape(np.fromfile(file_path, dtype=np.float32),(-1,3))

def load_points_sift(file_path):
    return np.reshape(np.fromfile(file_path, dtype=np.uint8),(-1,128))

def load_depth_map(file_path,dtype=np.float16):
    with open(file_path,'rb') as f:
        fbytes = f.read()
    w,h,c=[int(x) for x in str(fbytes[:20])[2:].split('&')[:3]]
    header='{}&{}&{}&'.format(w,h,c)
    body=fbytes[len(header):]
    img=np.fromstring(body,dtype=dtype).reshape((h,w,c))
    return np.nan_to_num(img)

def load_image(file_path):
    return imread(file_path).astype(np.float32)

# Multi-matrix logical AND
def logical_and(mats):
    out = mats[0]
    for mat in mats[1:]:
        out = np.logical_and(out,mat)
    return out

# Compute scale & crop corners
def get_scale_and_crop_corners(img_h,img_w,scale_size,crop_size):
    sc = float(scale_size)/float(min(img_h,img_w))
    h = int(np.ceil(img_h*sc))
    w = int(np.ceil(img_w*sc))
    y0 = (h-crop_size)//2
    x0 = (w-crop_size)//2
    y1 = y0+crop_size
    x1 = x0+crop_size
    cc = [x0,x1,y0,y1]
    return sc, cc, h, w

# scale and crop image
def scale_crop(img,scale_size,crop_size,is_depth=False):    
    sc,cc,h,w = get_scale_and_crop_corners(img.shape[0],img.shape[1],scale_size,crop_size)
    x0, x1, y0, y1 = cc
    img = resize(img, (h,w), anti_aliasing=True, mode='reflect', preserve_range=True)
    img = img[y0:y1,x0:x1]
    if is_depth:
        img *= sc
    return img

# compute sudo ground-truth visibility map
def compute_visib_map(gt_depth,proj_depth,pct_diff_thresh=5.):
    is_val = logical_and([proj_depth > 0., gt_depth > 0.,
                          np.logical_not(np.isnan(proj_depth)), np.logical_not(np.isnan(gt_depth))])
    is_val = is_val.astype(np.float32)
    pct_diff = (proj_depth - gt_depth) / (gt_depth + 1e-8) * 100.
    pct_diff[np.isnan(pct_diff)] = 100.
    is_vis = (pct_diff < pct_diff_thresh).astype(np.float32) * is_val
    return is_vis, is_val
    
################################################################################
# Compute 2D projection of point cloud
################################################################################

# Compute 2D projection of point cloud
def project_points(pcl_xyz, pcl_rgb, pcl_sift, proj_mat, src_img_h, src_img_w, scale_size, crop_size):
    sc, cc, h, w = get_scale_and_crop_corners(src_img_h,src_img_w,scale_size,crop_size)
    x0, x1, y0, y1 = cc
    
    # Project point cloud to camera view & scale
    world_xyz = np.hstack((pcl_xyz,np.ones((len(pcl_xyz),1))))
    proj_xyz = (proj_mat.dot(world_xyz.T)).T
    proj_xyz[:,:2] = proj_xyz[:,:2] / proj_xyz[:,2:3]
    
    # scale point cloud
    x = np.rint(proj_xyz[:,0]*sc).astype(int)
    y = np.rint(proj_xyz[:,1]*sc).astype(int)
    z = proj_xyz[:,2]*sc

    # crop point cloud and filter out pts with invalid depths
    mask = logical_and([x>=x0, x<x1, y>=y0, y<y1, z>0., np.logical_not(np.isnan(z))])
    x=x[mask]; y=y[mask]; z=z[mask]

    # z-buffer xy coordinates with multiple descriptors
    idx = np.argsort(z)
    idx = idx[np.unique(np.ravel_multi_index((y,x), (h,w)),return_index=True)[1]]
    x = x[idx]-x0
    y = y[idx]-y0
    
    # get projected point cloud scaled & cropped
    proj_depth = np.zeros((crop_size,crop_size,1)).astype(np.float32)
    proj_rgb = np.zeros((crop_size,crop_size,3)).astype(np.uint8)
    proj_sift = np.zeros((crop_size,crop_size,128)).astype(np.uint8)
    proj_depth[y,x] = z[idx,None]
    proj_rgb[y,x] = pcl_rgb[mask][idx]
    proj_sift[y,x] = pcl_sift[mask][idx]

    return proj_depth, proj_rgb, proj_sift
