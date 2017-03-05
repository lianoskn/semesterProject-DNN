
import os
from scipy import misc
import numpy as np
from random import shuffle


"""Class that parses a directory of images, under the specific format, 
and provides with single images or batches of images. Performs data augmentation 
by random jumps over the dataset, random jittering between 0 and 4 pixels and randomly flipping 
horizontally.
For the Optical flow dataset, the isComposite option must be set to True.

Assumes:
        that the data are inside the path dirpath/datapath/..
        that there exists the file mean_img.png inside dirpath/
        
 """
class ImageReader:
        
    def __init__(self,dirpath,datapath,istest,imdim=64,isComposite = False):
        self._index = 0
        self._dirpath = dirpath + datapath
        
        self.mean_img = misc.imread(dirpath+'train_mean.png').astype(np.float32)

        self._iscomposite = isComposite

        self._epochs = 0
        self._istest = istest   
        np.random.seed()     
        self._epochIncr = 0
       
        self._fileslist = next(os.walk(self._dirpath))[2]
        self._dirsize = len(self._fileslist)
        
        self._imdim = imdim
        self._imsize = imdim*imdim
        
    def read_batch(self,batchsize):
        
        imsequence = np.zeros([batchsize, self._imsize])
        if self._iscomposite:
            imsequence = np.zeros([batchsize, self._imsize,3])
            
        labelsequence = np.zeros([batchsize,2])
        
        for i in range(batchsize):
            jitt = -1
            flip = np.round(np.random.random()-0.2)
            
            im,lbl = self.read_single_file(flip,jitt)
            
            imsequence[i,:] = im
            labelsequence[i,lbl] = 1

        if self._epochIncr == 1:
            self._epochIncr = 0
            shuffle(self._fileslist)
            self._index = 0

        return imsequence,labelsequence        


    def invoke_thread(self):
        True

    """Reads a single file from the list, incrementing the index"""
    """Arguments:
        flip: if true, then flip the image left-to-right
        
        jitt: if -1,       then randomly displace the image by [0,4]
              if in [0,4], then displace by that amount
              if > 4,      no displacement
    """
    def read_single_file(self,flip,jitt = 0):
        file = self._fileslist[self._index]
        img = misc.imread(self._dirpath+file)
        
        imdim = self._imdim 
        self._index = (self._index + 1) % (self._dirsize)
        if self._index == 0:
            print("Epoch complete")
            self._epochs += 1
            self._epochIncr = 1
            
        # get input image
        img.astype(np.float32)

        if not self._iscomposite:        
            img = img - self.mean_img 
        else:
            img[:,:,2] = img[:,:,2] - self.mean_img[:,:,2] 
    
        #parse tokens
        tokens = file.split("_")
        tokens[len(tokens)-1] = tokens[len(tokens)-1].split(".")[0]    
        
        imlabel = tokens[4]  
       
        if flip:
            a = np.copy(img)
            for i in range(64):
                a[:,i] = img[:,-i-1]
            img = a
        
        b = np.copy(img)
        #jitter random
        if (not self._istest) and (jitt < 5):
            q = np.floor(np.random.rand()*4.9)
            if jitt >= 0:
                q = jitt
            a = img[q: 64-(4-q), q: 64-(4-q)]
            b=np.pad(a,(2,2),'edge')
        
        #img = b

        #crop it
        q = 64 - imdim
    
        if q > 0:
            img = b[q/2: 64-q/2, q/2: 64-q/2]
        
        if self._iscomposite:
            img = np.reshape(np.copy(img),[1,imdim*imdim,3])
        else:
            img = np.reshape(np.copy(img),imdim*imdim)

    
        return img,imlabel  

    