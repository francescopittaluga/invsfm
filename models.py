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
# models.py
# Contains all model definitions
# Author: Francesco Pittaluga

import tensorflow as tf
import numpy as np

# Base Model
class Net(object):

    def __init__(self):
        self.weights = {}
    
    # Save weights to an npz file
    def save(self,sess,fname):
        wts = {k:sess.run(v) for k,v in self.weights.items()}
        np.savez(fname,**wts)
        return wts
    
    # Load weights from an npz file
    def load(self,sess,fname=None):
        wts = np.load(fname)
        ops = [v.assign(wts[k].astype(np.float32)).op
               for k,v in self.weights.items() if k in wts]
        if len(ops) > 0:
            sess.run(ops)

    # Get all trainable weights
    def trainable_variables(self):  
        return {v:k for k,v in self.weights.items() if v in tf.trainable_variables()}

            
# Base Model for VisibNet, CaarseNet and RefineNet
class InvNet(Net):
    def __init__(self, inp,
                 bn='train',
                 ech = [256,256,256,512,512,512],
                 dch = [512,512,512,256,256,256,128,64,32,3],
                 skip_conn = 6,
                 conv_act = 'relu',
                 outp_act = 'tanh'):

        super().__init__()
        self.bn = bn
        self.weights = {}
        self.ifdo = tf.Variable(False,dtype=tf.bool)
        self.set_ifdo = self.ifdo.assign(True).op
        self.unset_ifdo = self.ifdo.assign(False).op
        
        #Encoder
        out = inp; skip = [out]
        for i in range(len(ech)):
            out = self.conv(out,4,ech[i],2,True,1.,conv_act,'ec%d'%i)
            skip.append(out)
        skip = list(reversed(skip))[1:]
                
        # Decoder
        for i in range(len(dch)-1):
            if i<len(ech): out = tf.image.resize_images(out,tf.shape(out)[1:3]*2,method=1)
            out = self.conv(out,3,dch[i],1,True,.5 if i<3 else 1.,conv_act,'dc%d'%i)
            if i<skip_conn: out = tf.concat((skip[i],out),axis=3) 
        self.pred = self.conv(out,3,dch[-1],1,False,1.,outp_act,'dc%d'%(len(dch)-1))

        
    # Covolutional layer with Batchnorm, Bias, Dropout  & Activation
    def conv(self,inp,ksz,nch,stride,bn,rate,act,nm):

        # Conv
        ksz = [ksz,ksz,inp.get_shape().as_list()[-1],nch]
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        self.weights['%s_w'%nm] = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        out = tf.pad(inp,[[0,0],[1,1],[1,1],[0,0]],'REFLECT')
        out = tf.nn.conv2d(out,self.weights['%s_w'%nm],[1,stride,stride,1],'VALID')

        # Batchnorm
        if bn:
            if self.bn=='train' or self.bn=='set':
                axis = list(range(len(out.get_shape().as_list())-1))
                wmn = tf.reduce_mean(out,axis)
                wvr = tf.reduce_mean(tf.squared_difference(out,wmn),axis)
                out = tf.nn.batch_normalization(out,wmn,wvr,None,None,1e-3)

                if self.bn=='set':
                    self.weights['%s_mn'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
                    self.weights['%s_vr'%nm] = tf.Variable(tf.ones([nch],dtype=tf.float32))
                    self.bn_outs['%s_mn'%nm] = wmn
                    self.bn_outs['%s_vr'%nm] = wvr
                    
            if self.bn=='test':
                self.weights['%s_mn'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
                self.weights['%s_vr'%nm] = tf.Variable(tf.ones([nch],dtype=tf.float32))
                out = tf.nn.batch_normalization(out,self.weights['%s_mn'%nm],
                                                self.weights['%s_vr'%nm],None,None,1e-3)
                
        # Bias
        self.weights['%s_b'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
        out = out + self.weights['%s_b'%nm]

        # Dropout
        if rate < 1:
            out = tf.cond(self.ifdo, lambda: tf.nn.dropout(out,rate), lambda: out)
        
        # Activation
        if act=='relu':
            out = tf.nn.relu(out)
        elif act=='lrelu':
            out = tf.nn.leaky_relu(out)
        elif act=='sigm':
            out = tf.nn.sigmoid(out)
        elif act=='tanh':
            out = tf.nn.tanh(out)
            
        return out

    
# VisibNet 
class VisibNet(InvNet):
    
    def __init__(self,inp,bn='train',outp_act=True):

        if inp.get_shape().as_list()[-1] < 5:
            ech = [64,128,256,512,512,512]
        else:
            ech = [256,256,256,512,512,512]

        super().__init__(inp,bn=bn,
                         ech = ech,
                         dch = [512,512,512,256,256,256,128,64,32,1],
                         skip_conn = 6,
                         conv_act = 'relu',
                         outp_act = 'sigm' if outp_act else None)

# CoarseNet 
class CoarseNet(InvNet):
    
    def __init__(self,inp,bn='train',outp_act=True):

        super().__init__(inp,bn=bn,
                         ech = [256,256,256,512,512,512],
                         dch = [512,512,512,256,256,256,128,64,32,3],
                         skip_conn = 6,
                         conv_act = 'relu',
                         outp_act = 'tanh' if outp_act else None)

# RefineNet 
class RefineNet(InvNet):
    
    def __init__(self,inp,bn='train',outp_act=True):

        super().__init__(inp,bn=bn,
                         ech = [256,256,256,512,512,512],
                         dch = [512,512,512,256,256,256,128,64,32,3],
                         skip_conn = 4,
                         conv_act = 'lrelu',
                         outp_act = 'tanh' if outp_act else None)


# Convolutional layers of VGG16
class VGG16(Net):

    def __init__(self,inp,stop_layer=''):
        super().__init__()
        self.pred = {}

        # Subtract channel means
        mean = tf.constant([123.68,116.779,103.939], dtype=tf.float32, shape=[1,1,1,3])
        out = inp-mean

        # Set up convolution layers
        numl = [2,2,3,3,3]
        numc = [64,128,256,512,512]
        for i in range(len(numl)):
            for j in range(numl[i]):
                nm = 'conv{}_{}'.format(i+1,j+1) 
                if nm+'_W' not in self.weights:
                    ksz = [3,3,out.get_shape().as_list()[-1],numc[i]]
                    self.weights[nm+'_W']=tf.Variable(tf.truncated_normal(ksz,dtype=tf.float32,stddev=1e-1), trainable=False)
                    self.weights[nm+'_b']=tf.Variable(tf.zeros(numc[i], dtype=tf.float32), trainable=False)

                out = tf.nn.conv2d(out, self.weights[nm+'_W'], [1,1,1,1], padding='SAME')
                out = tf.nn.bias_add(out, self.weights[nm+'_b'])
                out = tf.nn.relu(out)
                self.pred[nm] = out
                if nm == stop_layer: 
                    return
    
            out = tf.nn.max_pool(out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            nm = 'pool{}'.format(i+1)
            self.pred[nm] = out
            if nm == stop_layer: 
                return


# Discriminator network for training RefineNet
class Discriminator(Net):

    def __init__(self):
        super().__init__()
        self.ifdo = tf.Variable(False,dtype=tf.bool)
        self.set_ifdo = self.ifdo.assign(True).op
        self.unset_ifdo = self.ifdo.assign(False).op
        

    def pred(self,inp):
        ncls = 2
        out = inp[0]

        # conv layers
        cch = [256, 256, 256, 512, 512]
        for i,ch in enumerate(cch):
            if i > 0 and i < len(inp):
                out = tf.concat((inp[i],out),axis=3)
            out = self.conv(out,3,ch,1,True,1,'lrelu','SAME','c%d'%i)
            out = tf.nn.max_pool(out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        # fc layers
        fch = [1024,1024,1024]
        for i,ch in enumerate(fch):
            out = self.conv(out,out.get_shape().as_list()[2],ch,1,False,.5,'lrelu','VALID','fc%d'%i)
        out = self.conv(out,out.get_shape().as_list()[2],ncls,1,False,1,False,'VALID','fc%d'%(len(fch)+1))

        return tf.reshape(out,[-1,ncls])


    # Covolutional layer with Batchnorm, Bias, Dropout  & Activation
    def conv(self,inp,ksz,nch,stride,bn,rate,act,pad,nm):

        # Conv
        ksz = [ksz,ksz,inp.get_shape().as_list()[-1],nch]
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        if '%s_w'%nm not in self.weights:
            self.weights['%s_w'%nm] = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        out = tf.nn.conv2d(inp,self.weights['%s_w'%nm],[1,stride,stride,1],pad)

        # Batchnorm
        if bn:
            axis = list(range(len(out.get_shape().as_list())-1))
            wmn = tf.reduce_mean(out,axis)
            wvr = tf.reduce_mean(tf.squared_difference(out,wmn),axis)
            out = tf.nn.batch_normalization(out,wmn,wvr,None,None,1e-3)

        # Bias
        if '%s_b'%nm not in self.weights:
            self.weights['%s_b'%nm] = tf.Variable(tf.zeros([nch],dtype=tf.float32))
        out = out + self.weights['%s_b'%nm]

        # Activation
        if act=='lrelu':
            out = tf.nn.leaky_relu(out)

        # Dropout
        if rate < 1:
            out = tf.cond(self.ifdo, lambda: tf.nn.dropout(out,rate), lambda: out)


        return out
