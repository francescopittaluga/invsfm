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
# utils.py
# Random utilty functions
# Author: Francesco Pittaluga

import sys
import os
import time
import re
from glob import glob
import numpy as np
import tensorflow as tf
import argparse

# Parser that prints help upon error
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

# Print to stdout with date/time
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(s) + "\n")
    sys.stdout.flush()

# Print to stderr with date/time
def eprint(s):
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(s) + "\n")
    sys.stderr.flush()

# Load annotations file
def load_annotations(fname):
    with open(fname,'r') as f:
        data = [line.strip().split(' ') for line in f]
    return np.array(data)

# Reading in batches (with repeatable random shuffling)
class batcher:
    def __init__(self,fname,bsz,niter=0):

        # Load from file
        self.data = load_annotations(fname)

        # Setup batching
        nsamp = len(self.data)
        self.bsz = bsz
        
        self.rand = np.random.RandomState(0)
        idx = self.rand.permutation(nsamp)
        for i in range(niter*bsz // len(idx)):
            idx = self.rand.permutation(len(idx))

        self.idx = np.int32(idx)
        self.pos = niter*bsz % len(self.idx)

    def get_batch(self):
        if self.pos+self.bsz >= len(self.idx):
            bidx = self.idx[self.pos:]

            idx = self.rand.permutation(len(self.idx))
            self.idx = np.int32(idx)

            self.pos = 0
            if len(bidx) < self.bsz:
                self.pos = self.bsz-len(bidx)
                bidx2 = self.idx[0:self.pos]
                bidx = np.concatenate((bidx,bidx2))
        else:
            bidx = self.idx[self.pos:self.pos+self.bsz]
            self.pos = self.pos+self.bsz

        return self.data[bidx]

# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])

                
# Save Optimizer state (Assume Adam)
def saveopts(fn,opts,others,sess):
    weights = {}
    for i in range(len(opts)):
        opt = opts[i][0]
        vdict = opts[i][1]
        if type(opt) == tf.train.AdamOptimizer:
            b1p, b2p = opt._get_beta_accumulators()
            weights['%d:b1p'%i] = b1p.eval(sess)
            weights['%d:b2p'%i] = b2p.eval(sess)
            for v in vdict.keys():
                nm = vdict[v]
                weights['%d:m_%s' % (i,nm)] = opt.get_slot(v,'m').eval(sess)
                weights['%d:v_%s' % (i,nm)] = opt.get_slot(v,'v').eval(sess)
            else:
                slots = opt.get_slot_names()
                for v in vdict.keys():
                    nm = vdict[v]
                    for s in slots:
                        weights['%d:%s%s' % (i,s,nm)] = opt.get_slot(v, s).eval(sess)
                        
    weights.update(others)
    np.savez(fn,**weights)
                        
                        
# Load Optimizer state (Assume Adam)
def loadopts(fn,opts,others,sess):
    if not os.path.isfile(fn):
        return None
    weights = np.load(fn)
    
    ph = tf.placeholder(tf.float32)
    for i in range(len(opts)):
        opt = opts[i][0]
        vdict = opts[i][1]
        
        if type(opt) == tf.train.AdamOptimizer:
            b1p, b2p = opt._get_beta_accumulators()
            sess.run(b1p.assign(ph),feed_dict={ph: weights['%d:b1p'%i]})
            sess.run(b2p.assign(ph),feed_dict={ph: weights['%d:b2p'%i]})
            for v in vdict.keys():
                nm = vdict[v]
                sess.run(opt.get_slot(v,'m').assign(ph),
                         feed_dict={ph: weights['%d:m_%s' % (i,nm)]})
                sess.run(opt.get_slot(v,'v').assign(ph),
                         feed_dict={ph: weights['%d:v_%s' % (i,nm)]})
            else:
                slots = opt.get_slot_names()
                for v in vdict.keys():
                    nm = vdict[v]
                    for s in slots:
                        sess.run(opt.get_slot(v, s).assign(ph),
                                 feed_dict={ph: weights['%d:%s%s' % (i,s,nm)]})
                        
    oval = [weights[k] for k in others]
    return oval
