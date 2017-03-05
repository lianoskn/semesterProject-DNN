
from myRNN import MyRNN
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell
from numpy import inf


class MyFusionRNN(MyRNN):
        
    def __init__(self,nclasses,rnnsize,batchsize,maxseq,baseCNN1,baseCNN2, inputdim,nlayers = 1):
        
        super().__init__(nclasses,rnnsize,batchsize,maxseq,baseCNN1,inputdim,nlayers = 1)  
                       
        self._baseCNN2 = baseCNN2
        #self.x2 = tf.placeholder(tf.float32, [None,4096])

    def inference(self):
        
        if self._inputdim == 64*64: 
            inputs = self.x
        else:
            #split vertically
            x1,x2 = tf.split(0,2,self.x)
            pred,conv_end,d1,d2,d3 = self._baseCNN.inference(x1, self._baseCNN.weights, self._baseCNN.biases, self.keep_prob)   
            pred2,conv_end2,d1_2,d2_2,d3_2 = self._baseCNN2.inference(x2, self._baseCNN2.weights, self._baseCNN2.biases, self.keep_prob)   
            if (self._inputdim == 2*400):
                inputs = tf.concat(1, [d1,d1_2])
            elif self._inputdim == 128*64:
                inputs = conv_end

        inputs = tf.nn.dropout(inputs, self.keep_prob)
        
        inputs = [tf.reshape(i, (self._batchsize, self._inputdim)) for i in tf.split(0, self._maxseq, inputs)]
            
        outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=self._initial_state, 
                                   sequence_length=self.early_stop)
        
        return outputs
    
    """Methods: calc_cost, predict, weighting same as parent"""
    
    
    
    