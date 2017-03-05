
from imageReader import ImageReader
import numpy as np

"""Child class of ImageReader, to read sequences of images. Exposes the same interface as ImageReader class.
In the constructor, the maximum length of the sequences must be provided. The, the sequences are subsampled accordingly.
"""
class SequenceReader(ImageReader):

    def __init__(self,dirpath,datapath,istest,maxsequence,imdim = 64,isComposite = False):
        super().__init__( dirpath,datapath,istest,imdim,isComposite)
        self._fileslist.sort()
        self._maxseq = maxsequence
        
        #will ignore sequences with length < _ignoresmall
        self._ignoresmall = 1
        
        """Returns a batch of sequences, with even positive and negative class sequences.
         Returns: sequencelength: array of length = batchsize, with the subsampled lengths of the sequences.
                   labelsequence: array of labels for each sequence of the batch  
                      imsequence: Batch of image sequences. size: maxsequence*batchsize X imagesize
                                  outer_idx: maxseq, inner_idx: batchsize
                """
    def read_batch(self,batchsize):
        
        imsequence = np.zeros([batchsize*self._maxseq,self._imsize])
        if self._iscomposite:
            imsequence = np.zeros([batchsize*self._maxseq,self._imsize,3])
        labelsequence = np.zeros([batchsize,self._maxseq])
        sequencelength =[]
        """imsequence: outer_idx: maxseq, inner_idx: batchsize"""
        """labelsequence: batch_id X maxseq"""
        i = 0
        c = np.zeros([2,])
        c[1] = np.ceil(batchsize/2.)
        c[0] = np.floor(batchsize/2.)
        while i < batchsize:
            ids = [j for j in range(i,batchsize*self._maxseq,batchsize)]
            seqlen,im,lbl = self.read_sequence()
            """ignore small sequences!!"""
            if seqlen < self._ignoresmall:
                continue
            
            """get batches with equal positive and negative class samples only when training"""
            if (not self._istest) and (batchsize > 1):
                if c[lbl[0]] == 0:
                    continue
                else:
                    c[lbl[0]] = c[lbl[0]] - 1
            imsequence[ids,:] = im
            labelsequence[i] = lbl
            sequencelength.append(seqlen)
            i = i + 1

        return sequencelength,imsequence,labelsequence        
        
    def read_sequence(self):
        while True:
            file = self._fileslist[self._index]
            tokens = file.split("_")
            tokens[len(tokens)-1] = tokens[len(tokens)-1].split(".")[0]    
            seqLen  = int(tokens[3])
            pivot_name = tokens[0]
            pivot_seqid = tokens[1]
            
            if self._istest:
                break
            p = np.random.rand()
            if p > 0.4:
                break
            #progress to the next sequence
            for i in range (0,seqLen+1):            
                file2 = self._fileslist[(self._index+i)%self._dirsize]
                testtokens = file2.split("_")
                testtokens[len(tokens)-1] = testtokens[len(tokens)-1].split(".")[0]    
                if pivot_name != testtokens[0] or pivot_seqid != testtokens[1]:
                    break        
            self._index = (self._index +i) % self._dirsize
        
        #read the pivot sequence
        
        imseq = np.zeros([seqLen,self._imsize])
        if self._iscomposite:
            imseq = np.zeros([seqLen,self._imsize,3])
        labelseq = np.zeros([seqLen])
        
        #randomly flip and crop the whole sequence
        flipall = 0
        impadding = 0
        if not self._istest:
            impadding = np.floor(np.random.rand()*4.9) 
            p = np.random.rand()
            if p < 0.35:
                flipall = 1
    
        ind = []
        
        for i in range (0,seqLen+1):
            file = self._fileslist[self._index]
            tokens = file.split("_")
            tokens[len(tokens)-1] = tokens[len(tokens)-1].split(".")[0]    
            if pivot_name == tokens[0] and pivot_seqid == tokens[1]:
                im,lbl = self.read_single_file(flipall,impadding) # increments the index
                imseq[i,:] = im
                labelseq[i] = lbl
                ind.append(int(tokens[2]))
            else:
                break
        
        self.trueseqlen = seqLen 
        seqLen = i
        #indices ind are like 0,1,10,11,..,2,20,..
        #    so sort them
        sort_index = np.argsort(ind)     
        labelseq = np.copy(labelseq[sort_index])
        imseq = np.copy(imseq[sort_index,:])
        
        imsequence = np.zeros([self._maxseq,self._imsize])
        if self._iscomposite:
            imsequence = np.zeros([self._maxseq,self._imsize,3])

        labelsequence = np.zeros([self._maxseq])
        
        if seqLen == self._maxseq:
            imsequence = imseq
            labelsequence = labelseq
        if seqLen < self._maxseq:
            #padd 
            imsequence[0:seqLen,:] = imseq[0:seqLen,:]
            labelsequence[0:seqLen] = labelseq[0:seqLen]            
            imsequence[seqLen::,:] = 0.#imseq[0,:]
            labelsequence[seqLen::] = -1
        if seqLen > self._maxseq:
            #sample   
            ids = self.seq_sampling(seqLen)
            #print(seqLen)
            seqLen = len(ids)     
            imsequence[0:seqLen,:] = np.copy(imseq[ids,:])
            labelsequence[0:seqLen] = np.copy(labelseq[ids])
            imsequence[seqLen::,:] = 0.#imseq[0,:]
            labelsequence[seqLen::] = -1  

        return seqLen,imsequence,labelsequence           
    
    
    def seq_sampling(self,seqlen):
        mseq = self._maxseq
        sampling_rate = (seqlen/mseq)+.0000001
        c = [int(np.round(i)) for i in np.arange(0,seqlen,sampling_rate)]
        return c
    
            
