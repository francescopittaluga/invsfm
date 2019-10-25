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
# demo_5k.py
# Demo script for running pre-trained models on infsfm data
# Author: Francesco Pittaluga

import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import utils as ut
import load_data as ld
from models import VisibNet
from models import CoarseNet
from models import RefineNet

################################################################################

parser = ut.MyParser(description='Configure')
parser.add_argument("--input_attr", type=str, default='depth_sift_rgb',
                    choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
                    help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")
parser.add_argument("--pct_3D_points", type=float, default=100., choices=[20,60,100],
                    help="%(type)s: Percent of available 3D points to include in input tensor (default: %(default)s)")
parser.add_argument("--crop_size", type=int, default=512, choices=[256,512],
                    help="%(type)s: Size to crop images to (default: %(default)s)")
parser.add_argument("--scale_size", type=int, default=512, choices=[256,394,512],
                    help="%(type)s: Size to scale images to before crop (default: %(default)s)")
parser.add_argument("--num_samples", type=int, default=32,
                    help="%(type)s: Number of samples to process/visualize (default: %(default)s)")
parser.add_argument("--seed", type=int, default=1111,
                    help="%(type)s: Seed for random selection of samples (default: %(default)s)")
prm = parser.parse_args()

if prm.scale_size < prm.crop_size: parser.error("SCALE_SIZE must be >= CROP_SIZE")
if prm.num_samples <= 0: parser.error("NUM_SAMPLES must be > 0")

prm_str = 'Arguments:\n'+'\n'.join(['{} {}'.format(k.upper(),v) for k,v in vars(prm).items()])
print(prm_str+'\n')

# set paths for model wts
vnet_wts_fp = 'wts/pretrained/{}/visibnet.model.npz'.format(prm.input_attr)
cnet_wts_fp = 'wts/pretrained/{}/coarsenet.model.npz'.format(prm.input_attr)
rnet_wts_fp = 'wts/pretrained/{}/refinenet.model.npz'.format(prm.input_attr)

################################################################################

# Load annotations
anns = ut.load_annotations('data/anns/demo_5k/test.txt')
anns = anns[np.random.RandomState(seed=prm.seed).permutation(len(anns))]

# Load data
proj_depth = []
proj_sift = [] 
proj_rgb = []
src_img = []
gt_vis = []
for i in range(prm.num_samples):
    
    # Load point cloud
    pcl_xyz = ld.load_points_xyz('data/'+anns[i,0])
    pcl_rgb = ld.load_points_rgb('data/'+anns[i,1])
    pcl_sift = ld.load_points_sift('data/'+anns[i,2])
    
    # Load camera 
    K,R,T,h,w = ld.load_camera('data/'+anns[i,3])
    proj_mat = K.dot(np.hstack((R,T)))
    
    # Project point cloud to camera
    pdepth, prgb, psift = ld.project_points(pcl_xyz, pcl_rgb, pcl_sift,
                                            proj_mat, h, w, prm.scale_size, prm.crop_size)
    simg = ld.scale_crop(ld.load_image('data/'+anns[i,4])/127.5-1.,prm.scale_size,prm.crop_size)
    gt_depth = ld.scale_crop(ld.load_depth_map('data/'+anns[i,5],dtype=np.float16).astype(np.float32),
                             prm.scale_size,prm.crop_size,is_depth=True)
    is_vis, is_val = ld.compute_visib_map(gt_depth,pdepth,pct_diff_thresh=5.)  
    
    proj_depth.append((pdepth*is_val)[None,...])
    proj_sift.append((psift*is_val)[None,...])
    proj_rgb.append((prgb*is_val)[None,...])
    src_img.append(simg[None,...])
    gt_vis.append(is_vis[None,...])
    
proj_depth = np.vstack(proj_depth)
proj_sift = np.vstack(proj_sift)
proj_rgb = np.vstack(proj_rgb)
src_img = np.vstack(src_img)
gt_vis = np.vstack(gt_vis)

################################################################################
# Build Graph

proj_depth_p = tf.placeholder(tf.float32,shape=[1,prm.crop_size,prm.crop_size,1])
proj_rgb_p = tf.placeholder(tf.uint8,shape=[1,prm.crop_size,prm.crop_size,3])
proj_sift_p = tf.placeholder(tf.uint8,shape=[1,prm.crop_size,prm.crop_size,128])

pdepth = proj_depth_p
prgb = tf.to_float(proj_rgb_p)
psift = tf.to_float(proj_sift_p)

keep = prm.pct_3D_points/100.
pdepth = tf.nn.dropout(pdepth,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
prgb = tf.nn.dropout(prgb,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
psift = tf.nn.dropout(psift,keep,noise_shape=[1,prm.crop_size,prm.crop_size,1],seed=0)*keep
valid = tf.greater(pdepth,0.)

# set up visibnet
if prm.input_attr=='depth':
    vinp = pdepth
elif prm.input_attr=='depth_rgb':
    vinp = tf.concat((pdepth, prgb/127.5-1.),axis=3)
elif prm.input_attr=='depth_sift':
    vinp = tf.concat((pdepth, psift/127.5-1.),axis=3)
elif prm.input_attr=='depth_sift_rgb':
    vinp = tf.concat((pdepth, psift/127.5-1., prgb/127.5-1.),axis=3)                     
vnet = VisibNet(vinp,bn='test')
vpred = tf.logical_and(tf.greater(vnet.pred,.5),valid)
vpredf = tf.to_float(vpred)*0.+1.

# set up coarsenet 
if prm.input_attr=='depth':
    cinp = pdepth*vpredf
elif prm.input_attr=='depth_rgb':
    cinp = tf.concat((pdepth*vpredf, prgb*vpredf/127.5-1.),axis=3)
elif prm.input_attr=='depth_sift':
    cinp = tf.concat((pdepth*vpredf, psift*vpredf/127.5-1.),axis=3)
elif prm.input_attr=='depth_sift_rgb':
    cinp = tf.concat((pdepth*vpredf, psift*vpredf/127.5-1., prgb*vpredf/127.5-1.),axis=3)
cnet = CoarseNet(cinp,bn='test')
cpred = cnet.pred

# set up refinenet
rinp = tf.concat((cpred,cinp),axis=3)
rnet = RefineNet(rinp,bn='train')
rpred = rnet.pred

# scale outputs
cpred = (cpred+1.)*127.5
rpred = (rpred+1.)*127.5

################################################################################

# Run Graph
sess=tf.Session()
try: init_all_vars = tf.global_variables_initializer()
except: init_all_vars = tf.initialize_all_variables()

# Load net wts
vnet.load(sess,vnet_wts_fp)
cnet.load(sess,cnet_wts_fp)
rnet.load(sess,rnet_wts_fp)
sess.run([vnet.unset_ifdo,
          cnet.unset_ifdo,
          rnet.unset_ifdo])

# Run cnet
vpred_img = []
cpred_img = []
rpred_img = []
valid_img = []
for i in range(prm.num_samples):
    fd = {proj_depth_p:proj_depth[i:i+1],
          proj_rgb_p:proj_rgb[i:i+1],
          proj_sift_p:proj_sift[i:i+1]}
    out = sess.run([vpred,cpred,rpred,valid],feed_dict=fd)
    vpred_img.append(out[0])
    cpred_img.append(out[1])
    rpred_img.append(out[2])
    valid_img.append(out[3])
vpred_img = np.vstack(vpred_img)
cpred_img = np.vstack(cpred_img)
rpred_img = np.vstack(rpred_img)
valid_img = np.vstack(valid_img)
        
################################################################################

# Generate visibnet visualization
vpred = np.vstack(vpred_img)
valid = np.vstack(valid_img)
zero = np.zeros(valid.shape,dtype=bool)
vpred_img = np.ones([vpred.shape[0],prm.crop_size,3])*255.
vpred_img[np.dstack((valid,valid,valid))] = 0.
vpred_img[np.dstack((np.logical_and(valid,np.logical_not(vpred)),zero,zero))] = 255.
vpred_img[np.dstack((zero,zero,np.logical_and(valid,vpred)))] = 255.

# Generate gt visibility map visualization
visib = np.vstack(gt_vis)
gt_vis = np.ones([visib.shape[0],prm.crop_size,3])*255.
gt_vis[np.dstack((valid,valid,valid))] = 0.
gt_vis[np.dstack((np.logical_and(valid,np.logical_not(visib)),zero,zero))] = 255.
gt_vis[np.dstack((zero,zero,np.logical_and(valid,visib)))] = 255.

# Build results montage
border_size = 25
header_size = 60
mntg = np.hstack((np.vstack((src_img+1.)*127.5).astype(np.uint8),
                  gt_vis.astype(np.uint8),
                  np.zeros((gt_vis.shape[0],border_size,3)).astype(np.uint8),
                  vpred_img.astype(np.uint8),
                  np.vstack(cpred_img).astype(np.uint8),
                  np.vstack(rpred_img).astype(np.uint8)))
header_bot = np.ones((header_size,mntg.shape[1],3))*127.
header_top = np.zeros((header_size,mntg.shape[1],3))
mntg = np.vstack((header_top,header_bot,mntg))

# Add titles to mntg header
mntg = Image.fromarray(mntg.astype(np.uint8))
im_draw = ImageDraw.Draw(mntg)
font = ImageFont.truetype("FreeMonoBold.ttf", 36)
column_titles = ['Target Image','Pseudo-GT Visibility','VisibNet Prediction',
                 'CoarseNet Prediction','RefineNet Prediction']
figure_title = 'Input Attributes: ' + prm.input_attr.replace('_',', ')
for i in range(len(column_titles)):
    xpos = prm.crop_size*i + prm.crop_size/2 - font.getsize(column_titles[i])[0]/2
    im_draw.text((xpos,70), column_titles[i], font=font, fill=(255,255,255))
xpos = header_top.shape[1]/2-font.getsize(figure_title)[0]/2
im_draw.text((xpos,10), figure_title, font=font, fill=(255,255,255))

# save montage
fp = 'viz/demo_5k/{}.png'.format(prm.input_attr)
print('Saving visualization to {}...'.format(fp))
mntg.save(fp)
print('Done')

