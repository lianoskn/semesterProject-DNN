
from myCNN import CNN_sequence_predict
import tensorflow as tf
import numpy as np
from tensorNet import TensorNet


class MyFusionCNN(TensorNet):
        
    def __init__(self,nclasses,baseCNN1,baseCNN2, inputdim, istrainable=True):
        
        super().__init__(nclasses)  
                       
        self.y = tf.placeholder(tf.float32, [None,2])

        self._baseCNN = baseCNN1
        self._baseCNN2 = baseCNN2
        
        self._inputdim = inputdim
        #self.x2 = tf.placeholder(tf.float32, [None,4096])
        self._istrainable = istrainable
        
        self.initializevars()
        
        self.inference()


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
                    
        projlayer = tf.matmul(inputs, self.weights['wc1']) + self.biases['wc1']
        self.projlayer = tf.nn.dropout(projlayer,self.keep_prob)
        
        projlayer2 = tf.matmul(self.projlayer, self.weights['wc2']) + self.biases['wc2']
        self.projlayer2 = tf.nn.dropout(projlayer2,self.keep_prob)

        self.logits = tf.matmul(self.projlayer, self.weights['softmax']) + self.biases['softmax']

        self.predictions = tf.argmax(self.logits,1)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

        return  self.predictions, self.projlayer, self.projlayer2, self.logits,inputs
    
    def calc_cost(self):
        """evaluate accuracy on indiv. frames"""
        
        correct_pred = tf.equal(self.predictions, tf.argmax(self.y,1))
        batchaccuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    
        return self.cost,batchaccuracy, self.predictions
    
    
    def sequence_predict(self,sess,seq_xs,dropout,seqlen,seq_ys = []):
        seq_xs = np.vstack( [ seq_xs[0][0:seqlen[0]],seq_xs[1][0:seqlen[0]] ] )
        seq_ys = seq_ys[0,0:seqlen[0]]

        return CNN_sequence_predict(self,sess,seq_xs,dropout,seqlen,seq_ys)
    
    
    def initializevars(self):
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([self._inputdim, 200],stddev=0.01),trainable = self._istrainable),
            'wc2': tf.Variable(tf.random_normal([200, 200],stddev=0.01),trainable = self._istrainable),
            'softmax': tf.Variable(tf.random_normal([200, self._nclasses],stddev=0.01),trainable = self._istrainable)
        }
        self.biases = {
            'wc1': tf.Variable(tf.random_normal([200],stddev=0.01),trainable = self._istrainable),
            'wc2': tf.Variable(tf.random_normal([200],stddev=0.01),trainable = self._istrainable),
            'softmax': tf.Variable(tf.random_normal([self._nclasses],stddev=0.01),trainable = self._istrainable)
        }
    
    