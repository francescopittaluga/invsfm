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
# train_visib.py
# Training script for VisibNet
# Author: Francesco Pittaluga

import os
import sys
import tensorflow as tf
import numpy as np
import ctrlc
import utils as ut
import load_data_tflo as ld
from models import VisibNet

#########################################################################

parser = ut.MyParser(description='Configure')
parser.add_argument("-log_file", default=False, action='store_true', help="%(type)s: Print stdout and stderr to log and err files")
parser.add_argument("--input_attr", type=str, default='depth_sift_rgb', choices=['depth','depth_sift','depth_rgb','depth_sift_rgb'],
                    help="%(type)s: Per-point attributes to inlcude in input tensor (default: %(default)s)")
parser.add_argument("--trn_anns", type=str, default='data/anns/demo_5k/train.txt',
                    help="%(type)s: Path to annotation file for training samples (default: %(default)s)")
parser.add_argument("--val_anns", type=str, default='data/anns/demo_5k/val.txt',
                    help="%(type)s: Path to annotation file for validation samples (default: %(default)s)")
parser.add_argument("--batch_size", type=int, default=4, help="%(type)s: Number of images in batch (default: %(default)s)")
parser.add_argument("--crop_size", type=int, default=256, help="%(type)s: Size to crop images to (default: %(default)s)")
parser.add_argument("--scale_size", type=lambda s: [int(i) for i in s.split(',')], default=[296,394,512],
                    help="int,int,int: Sizes to randomly scale images to before cropping them (default: 296,394,512)")
parser.add_argument("--pct_3D_points", type=lambda s: [float(i) for i in s.split(',')][:2], default=[.05,1.0],
                    help="float,float: Min and max percent of 3D points to keep when performing random subsampling for data augmentation "+\
                    "(default: .05,1.0)")
parser.add_argument("--vis_thresh", type=float, default=.05, help="%(type)s: Threshold used to compute ground truth visibility mask."+\
                    "i.e., gt_visibibility_mask = ((inp_depth-gt_depth)/gt_depth) > VISIB_THRESH. (default: %(default)s)")
parser.add_argument("--max_iter", type=int, default=1e6, help="%(type)s: Stop training after MAX_ITER iterations (default: %(default)s)")
parser.add_argument("--log_freq", type=int, default=25, help="%(type)s: Log training stats every LOG_FREQ iterations (default: %(default)s)")
parser.add_argument("--chkpt_freq", type=int, default=1e4, help="%(type)s: Save model state every CHKPT_FREQ iterations. Previous model state "+\
                    "is deleted after each new save (default: %(default)s)")
parser.add_argument("--save_freq", type=int, default=5e4, help="%(type)s: Permanently save model state every SAVE_FREQ iterations "+\
                    "(default: %(default)s)")
parser.add_argument("--val_freq", type=int, default=5e3, help="%(type)s: Run validation loop every VAL_FREQ iterations (default: %(default)s)")
parser.add_argument("--val_iter", type=int, default=128, help="%(type)s: Number of validation samples per validation loop (default: %(default)s)")
parser.add_argument("--adam_eps", type=float, default=1e-8, help="%(type)s: Epsilon parameter for adam optimizer (default: %(default)s)")
parser.add_argument("--adam_mom", type=float, default=.9, help="%(type)s: Momentum parameter for adam optimizer (default: %(default)s)")
parser.add_argument("--adam_lr", type=float, default=1e-5, help="%(type)s: Learning rate parameter for adam optmizer (default: %(default)s)")
prm = parser.parse_args()

prm_str = 'Arguments:\n'+'\n'.join(['{} {}'.format(k.upper(),v) for k,v in vars(prm).items()])
print(prm_str+'\n')

#########################################################################

# Create exp dir if does not exist
exp_dir = 'wts/{}/visibnet'.format(prm.input_attr)
os.system('mkdir -p {}'.format(exp_dir))

# redirect stdout and stderr to log files
if prm.log_file:
    sys.stdout = open(exp_dir+'/train.log', 'a')
    sys.stderr = open(exp_dir+'/info.log', 'a')

# Check for saved weights & find iter
vsave = ut.ckpter(exp_dir+'/iter_*.vmodel.npz')
osave = ut.ckpter(exp_dir+'/iter_*.opt.npz')
vpath = lambda itr: '%s/iter_%07d.vmodel.npz'%(exp_dir,itr)
opath = lambda itr: '%s/iter_%07d.opt.npz'%(exp_dir,itr)
niter = vsave.iter

# Load annotations
ut.mprint("Loading annotations")
tbchr = ut.batcher(prm.trn_anns,prm.batch_size,niter)
vbchr = ut.batcher(prm.val_anns,prm.batch_size,niter)
ut.mprint("Done!")

#########################################################################

# Set up data fetch

camera_fps = [tf.placeholder(tf.string) for i in range(prm.batch_size)]
pts_xyz_fps = [tf.placeholder(tf.string) for i in range(prm.batch_size)]
pts_rgb_fps = [tf.placeholder(tf.string) for i in range(prm.batch_size)]
pts_sift_fps = [tf.placeholder(tf.string) for i in range(prm.batch_size)]
gt_depth_fps = [tf.placeholder(tf.string) for i in range(prm.batch_size)]
getfeed = lambda fps: \
          dict([(ph,'data/'+fps[i,3]) for i,ph in enumerate(camera_fps)]+\
               [(ph,'data/'+fps[i,0]) for i,ph in enumerate(pts_xyz_fps)]+\
               [(ph,'data/'+fps[i,2]) for i,ph in enumerate(pts_sift_fps)]+\
               [(ph,'data/'+fps[i,1]) for i,ph in enumerate(pts_rgb_fps)]+\
               [(ph,'data/'+fps[i,5]) for i,ph in enumerate(gt_depth_fps)])
gt_depth = ld.load_img_bch(gt_depth_fps,prm.crop_size,prm.scale_size,isval=False,binary=True)
proj_depth,proj_sift,proj_rgb = ld.load_proj_bch(camera_fps,pts_xyz_fps,pts_sift_fps,pts_rgb_fps,
                                                 prm.crop_size,prm.scale_size,isval=False)

pd_b=[]; ps_b=[]; pr_b=[]; is_visible=[]; is_valid=[]
keep_prob = tf.random_uniform([prm.batch_size],minval=prm.pct_3D_points[0]/100.,
                              maxval=prm.pct_3D_points[1]/100.,dtype=tf.float32,seed=niter)

for i in range(prm.batch_size):
    # Get visible points
    proj_is_val = tf.to_float(tf.greater(proj_depth[i], 0.))
    gt_is_val = tf.to_float(tf.greater(gt_depth[i], 0.))
    is_val = proj_is_val*gt_is_val
    pd = proj_depth[i]*is_val
    ps = proj_sift[i]*is_val
    pr = proj_rgb[i]*is_val
    pct_diff = (pd-gt_depth[i])/(gt_depth[i]+1e-8)
    is_vis = tf.to_float(tf.less(pct_diff,prm.vis_thresh))*is_val

    # dropout (1-keep_prob)% of projected pts
    pd = tf.nn.dropout(pd,keep_prob[i],noise_shape=[prm.crop_size,prm.crop_size,1],seed=niter)*keep_prob[i]
    ps = tf.nn.dropout(ps,keep_prob[i],noise_shape=[prm.crop_size,prm.crop_size,1],seed=niter)*keep_prob[i]
    pr = tf.nn.dropout(pr,keep_prob[i],noise_shape=[prm.crop_size,prm.crop_size,1],seed=niter)*keep_prob[i]
    is_vis = tf.nn.dropout(is_vis,keep_prob[i],noise_shape=[prm.crop_size,prm.crop_size,1],seed=niter)*keep_prob[i]
    is_val = tf.nn.dropout(is_val,keep_prob[i],noise_shape=[prm.crop_size,prm.crop_size,1],seed=niter)*keep_prob[i]

    pd_b.append(tf.reshape(pd,[1,prm.crop_size,prm.crop_size,1]))
    ps_b.append(tf.reshape(ps,[1,prm.crop_size,prm.crop_size,128]))
    pr_b.append(tf.reshape(pr,[1,prm.crop_size,prm.crop_size,3]))
    is_visible.append(tf.reshape(is_vis,[1,prm.crop_size,prm.crop_size,1]))
    is_valid.append(tf.reshape(is_val,[1,prm.crop_size,prm.crop_size,1]))
    
proj_depth = tf.concat(pd_b,axis=0)
proj_sift = tf.concat(ps_b,axis=0) / 127.5 - 1.
proj_rgb = tf.concat(pr_b,axis=0) / 127.5 - 1.
is_visible = tf.concat(is_visible,axis=0)
is_valid = tf.concat(is_valid,axis=0)

#########################################################################

if prm.input_attr=='depth':
    vinp = proj_depth
    vinp_sz = [prm.batch_size,prm.crop_size,prm.crop_size,1]
elif prm.input_attr=='depth_sift':
    vinp = tf.concat((proj_depth,proj_sift),axis=3)
    vinp_sz = [prm.batch_size,prm.crop_size,prm.crop_size,129]
elif prm.input_attr=='depth_rgb':
    vinp = tf.concat((proj_depth,proj_rgb),axis=3)
    vinp_sz = [prm.batch_size,prm.crop_size,prm.crop_size,4]
elif prm.input_attr=='depth_sift_rgb':
    vinp = tf.concat((proj_depth,proj_rgb,proj_sift),axis=3)
    vinp_sz = [prm.batch_size,prm.crop_size,prm.crop_size,132]
    
# Set up pre-fetching
vinp_b0 = tf.Variable(tf.zeros(vinp_sz,dtype=tf.float32))
vinp_b1 = tf.Variable(tf.zeros(vinp_sz,dtype=tf.float32))

vgt = tf.concat([is_visible,is_valid],axis=3)
vgt_sz = [prm.batch_size,prm.crop_size,prm.crop_size,2]
vgt_b0 = tf.Variable(tf.zeros(vgt_sz,dtype=tf.float32))
vgt_b1 = tf.Variable(tf.zeros(vgt_sz,dtype=tf.float32))

tldr_fetchOp = [vinp_b0.assign(vinp).op, vgt_b0.assign(vgt).op]
vldr_fetchOp = [vinp_b1.assign(vinp).op, vgt_b1.assign(vgt).op]
tldr_swapOp = [vinp_b1.assign(vinp_b0).op, vgt_b1.assign(vgt_b0).op]

# Init coarse inverter
V = VisibNet(vinp_b1,bn='train',outp_act=False)
vpred = V.pred

#########################################################################

# Set optimizer
vvars = V.trainable_variables()
optV = tf.train.AdamOptimizer(prm.adam_lr,prm.adam_mom,epsilon=prm.adam_eps)

mask = tf.reshape(vgt_b1[:,:,:,1],[-1,1])
logs = tf.boolean_mask(tf.reshape(vpred,[-1,1]),mask)
lbls = tf.boolean_mask(tf.reshape(vgt_b1[:,:,:,0],[-1,1]),mask)
vloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=lbls,logits=logs))
vacc = tf.reduce_mean(tf.to_float(tf.equal(lbls,tf.to_float(tf.greater(tf.sigmoid(logs),0.5)))))
vStep = optV.minimize(vloss,var_list=list(vvars.keys()))

#########################################################################

# Start TF session (respecting OMP_NUM_THREADS)
try: init_all_vars = tf.global_variables_initializer()
except: init_all_vars = tf.initialize_all_variables()
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None: sess=tf.Session()
else: sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=int(nthr)))
sess.run(init_all_vars)

#########################################################################
# Load saved models & optimizers

# Load C wts
if vsave.latest != None:
    ut.mprint("Restoring V from " + vsave.latest )
    V.load(sess,vsave.latest)
    ut.mprint("Done!")
    
# Load optimizers
optlist = [[optV,vvars]]
if osave.latest is not None:
    ut.mprint("Restoring optimizers from " + osave.latest )
    ut.loadopts(osave.latest,optlist,[],sess)
    ut.mprint("Done!")

#########################################################################

# Main Training loop
sess.run(V.set_ifdo)
saviter = niter
tLossAcc=[]
vlog=''

fd=getfeed(tbchr.get_batch())
sess.run(tldr_fetchOp,feed_dict=fd)

ut.mprint("Starting from Iteration %d" % niter)
while not ctrlc.stop and niter < prm.max_iter:

    # Val loop
    if niter % prm.val_freq == 0:
        ut.mprint("Validating networks")
        sess.run(V.unset_ifdo)
        vLossAcc=[];
        for i in range(0,prm.val_iter):
            try: # prevent occasional failure when no pts in projection
                fps=vbchr.get_batch()
                fd=getfeed(fps)
                sess.run(vldr_fetchOp,feed_dict=fd)
                vLossAcc.append(sess.run([vloss,vacc]))
            except:
                pass
        sess.run(V.set_ifdo)
        args = list(np.mean(vLossAcc,axis=0))
        vlog=' val.loss {:.6f} val.acc {:.6f}'.format(*args)

    # Swap data buffers
    sess.run(tldr_swapOp)
    
    # Set up nxt data fetch op
    fps=tbchr.get_batch()
    fd=getfeed(fps)

    # Update vnet
    try: # prevent occasional failure when no pts in projection
        tLossAcc.append(sess.run([vloss,vacc,vStep]+tldr_fetchOp,feed_dict=fd)[:2])
    except:
        pass

    # Print training loss & accuracy
    niter+=1     
    if niter % prm.log_freq == 0:
        args = [niter]+list(np.mean(tLossAcc,axis=0))
        tlog = '[{:09d}] . trn.loss {:.6f} trn.acc {:.6f}'.format(*args)
        ut.mprint(tlog+vlog)
        tLossAcc=[]; vlog='';
        
    # Save models
    if niter % prm.chkpt_freq == 0:

        #Save VisibNet
        V.save(sess,vpath(niter))
        vsave.clean(every=prm.save_freq,last=1)
        ut.mprint("Saved weights to "+vpath(niter))

        # Save Optimizers
        ut.saveopts(opath(niter),optlist,{},sess)
        osave.clean(last=1)
        ut.mprint("Saved optimizers to "+opath(niter)) 
 
# Save models & optimizers
if niter > vsave.iter:
    
    # Save VisibNet
    V.save(sess,vpath(niter))
    vsave.clean(every=prm.save_freq,last=1)
    ut.mprint("Saved weights to "+vpath(niter))

    # Save Optimizers
    ut.saveopts(opath(niter),optlist,{},sess)
    osave.clean(last=1)
    ut.mprint("Saved optimizers to "+opath(niter)) 
