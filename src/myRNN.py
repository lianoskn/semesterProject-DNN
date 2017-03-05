
from tensorNet import TensorNet
import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn import rnn_cell
from numpy import inf


class MyRNN(TensorNet):
        
    def __init__(self,nclasses,rnnsize,batchsize,maxseq,baseCNN,inputdim,nlayers = 1):
        
        super().__init__(nclasses)  
        
        self.y = tf.placeholder(tf.int64, [batchsize, maxseq])
        
        self.cellsize = rnnsize
        self._batchsize = batchsize
        self._maxseq = maxseq
        self._inputdim = inputdim
        self._baseCNN = baseCNN
        self.numlayers = nlayers
        
        self.early_stop = tf.placeholder(tf.int32, [self._batchsize])
        
        self.cost_w = tf.placeholder(tf.float32, [self._batchsize,self._maxseq]) 
        
        initializer = tf.random_uniform_initializer(-.1,.1) 


        """Define the network """
        self.cell = tf.nn.rnn_cell.LSTMCell(self.cellsize, self._inputdim, initializer=initializer) 
        
        self.cell = rnn_cell.DropoutWrapper(
                                self.cell, output_keep_prob=self.keep_prob)
        
        if self.numlayers > 1:
            allcells = [self.cell]
            j = self.numlayers
            while j > 1:
                
                cell2 = tf.nn.rnn_cell.LSTMCell(self.cellsize, self.cellsize, initializer=initializer) 
                cell2 = rnn_cell.DropoutWrapper(
                                cell2, output_keep_prob=self.keep_prob)
                allcells.append(cell2)
                j = j - 1
            self.cell = rnn_cell.MultiRNNCell(allcells)

        """set initial state"""
        self._initial_state = self.cell.zero_state(self._batchsize, tf.float32)



    def inference(self):
        
        if self._inputdim == 64*64: 
            inputs = self.x
        else:
            pred,conv_end,d1,d2,d3 = self._baseCNN.inference(self.x, self._baseCNN.weights, self._baseCNN.biases, self.keep_prob)   
            if (self._inputdim == 400):
                inputs = d1
            elif self._inputdim == 128*64:
                inputs = conv_end

        inputs = tf.nn.dropout(inputs, self.keep_prob)
        
        inputs = [tf.reshape(i, (self._batchsize, self._inputdim)) for i in tf.split(0, self._maxseq, inputs)]
            
        outputs, state = tf.nn.rnn(self.cell, inputs, initial_state=self._initial_state, 
                                   sequence_length=self.early_stop)
        
        return outputs
    
    def calc_cost(self):
          
        output = self.inference()  
        output = tf.reshape(tf.concat(1, output), [-1, self.cellsize])
        """outputs: batchsize(outer index) * maxseq(inner index) X cellsize"""
        
        softmax_w = tf.get_variable("softmax_w", [self.cellsize, self._nclasses])
        softmax_b = tf.get_variable("softmax_b", [self._nclasses])
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        yy = tf.reshape(self.y, [-1])
        loss = seq2seq.sequence_loss_by_example(logits=[logits],
                                                targets=[yy],
                                                weights=[tf.reshape(self.cost_w, [-1])],
                                                average_across_timesteps=False)
        
        self.cost = tf.reduce_sum(loss) / self._batchsize   
        #self.cost =  tf.reduce_sum(loss) / tf.reduce_sum(self.cost_w)
        self.predictions = tf.argmax(logits,1)
        self.logits = logits
                
        return self.cost,self.predictions 
            
        
        """Make a single prediction per sequence of the batch"""
        """Input:
            sess: Tensorflow session object
            seq_xs: np.array, The batch of sequences. Dimensions: maxseq(outer) * batchsize(inner) X img_dim 
            seqlen: list, The length of each sequence in the batch. Dims: batchsize X 1 
            seq_ys: np.array, Optional, Dim Batchsize X maxseqlen. 
           Output: 
            sequence_prediction: np.array, predicted label for each sequence in the batch
            if seq_ys is provided:
                corr_preds: np.array, 1 if a sequence is predicted correctly
                cost: double, the perplexity
            """        
    def predict(self,sess,seq_xs,dropout,seqlen,w,seq_ys = []):
        
        cost = -inf
        if len(seq_ys) > 0:
            preds,cost,logs = sess.run( [self.predictions,self.cost,self.logits], feed_dict={self.x: seq_xs, self.y: seq_ys, 
                            self.keep_prob: dropout, self.early_stop: seqlen, self.cost_w: w})      
        else:
            preds,logs = sess.run( [self.predictions,self.logits], feed_dict={self.x: seq_xs, #self.y: seq_ys, 
                            self.keep_prob: dropout, self.early_stop: seqlen, self.cost_w: w})      
        
        #Reshape to [batchsize X maxseq]
        preds = np.reshape(preds,[self._batchsize,self._maxseq])
        
        l0 = np.reshape(logs[:,0],[self._batchsize,self._maxseq])
        l1 = np.reshape(logs[:,1],[self._batchsize,self._maxseq])

        sp0 = np.divide(np.sum(w*l0, 1),np.sum(w,1))
        sp1 = np.divide(np.sum(w*l1, 1),np.sum(w,1))
        soft_sequence_prediction = sp1 > sp0
        llogits = np.vstack([sp0,sp1])
        #sum per sequence
        
        sequence_prediction = np.divide(np.sum(w*preds, 1),np.sum(w,1))

        #find the sequences with equal votes (ambiguous)
        iseq = sequence_prediction == 1./2. 
        #set the prediction label for not ambiguous sequences
        sequence_prediction = sequence_prediction > 1./2.          
    
        # for the ambiguous, label the sequence according to the label of the last element
        if np.sum(iseq) > 0:
            preds = preds[iseq,:]
            seqlen = np.array(seqlen)
            sequence_prediction[iseq] = preds[:,seqlen[iseq]-1];
            
        #if seq_ys is provided, then output also correct predictions
        corr_preds = []
        if len(seq_ys) > 0:
            #corr_preds = seq_ys[:,0] == sequence_prediction
            corr_preds = seq_ys[:,0] == soft_sequence_prediction

        return soft_sequence_prediction, corr_preds, cost,llogits 
            
    """Creates weighting terms for prediction and loss"""
    """Should be called inside self.predict"""
    """input: length of each sequence in the batch to be processed """
    """Output: w [batchsize X maxseq] """        
    def weighting(self,seqlen,seqy = [],weights = 'uniform'):
        w=np.zeros([self._batchsize,self._maxseq])
        for i in range(0,self._batchsize):
            if weights == 'linear': 
                denom = np.sum(np.array([k for k in range(1,seqlen[i]+1)]))
            elif weights == 'square': 
                denom = np.sum(np.array([pow(k,2) for k in range(1,seqlen[i]+1)]))
            for j in range(0,self._maxseq):
                if weights == 'uniform':
                    w[i,j] = 1.*(j<seqlen[i])
                if weights == 'linear': 
                    w[i,j] = ((j+1)/denom)*seqlen[i]*(j<seqlen[i])    
                if weights == 'square': 
                    w[i,j] = (pow(j+1,2)/denom)*seqlen[i]*(j<seqlen[i])    
            if len(seqy) > 0:
                w[i,:] = w[i,:] - w[i,:]*(2./3.)*(seqy[i] == 0)		
        return w
        
"""Static method, same as above """        
def weighting(batchsize,maxseq,seqlen,seqy = [],weights = 'uniform'):
    w=np.zeros([batchsize,maxseq])
    for i in range(0,batchsize):
        if weights == 'linear': 
            denom = np.sum(np.array([k for k in range(1,seqlen[i]+1)]))
        elif weights == 'square': 
            denom = np.sum(np.array([pow(k,2) for k in range(1,seqlen[i]+1)]))
        for j in range(0,maxseq):
            if weights == 'uniform':
                w[i,j] = 1.*(j<seqlen[i])
            if weights == 'linear': 
                w[i,j] = ((j+1)/denom)*seqlen[i]*(j<seqlen[i])    
            if weights == 'square': 
                w[i,j] = (pow(j+1,2)/denom)*seqlen[i]*(j<seqlen[i])    
        if len(seqy) > 0:
            w[i,:] = w[i,:] - w[i,:]*(2./3.)*(seqy[i] == 0)        
    return w