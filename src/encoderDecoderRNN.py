import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from numpy import inf

class EncoderDecoderRNN():
        
    def createMultiLSTM(self,initializer):
        cell = tf.nn.rnn_cell.LSTMCell(self.cellsize, self._inputdim, initializer=initializer) 
        cell = rnn_cell.DropoutWrapper(
                                cell, output_keep_prob=self.keep_prob)    
        if self.numlayers > 1:
            allcells = [cell]
            j = self.numlayers
            while j > 1:
                
                cell2 = tf.nn.rnn_cell.LSTMCell(self.cellsize, self.cellsize, initializer=initializer) 
                cell2 = rnn_cell.DropoutWrapper(
                                cell2, output_keep_prob=self.keep_prob)
                allcells.append(cell2)
                j = j - 1
            cell = rnn_cell.MultiRNNCell(allcells)
        
        return cell
        
        
    def __init__(self,nclasses,rnnsize,batchsize,maxseq,inputdim,nlayers = 1,encoder_length = -1 ):
                
        #self.y = tf.placeholder(tf.int64, [batchsize, maxseq])
        #self.x = tf.placeholder(tf.float32, [batchsize*maxseq,inputdim])
         
        self.encoder_len = encoder_length
        if encoder_length == -1:
            self.encoder_len = maxseq/2
            
        self.encoder_inputs = tf.placeholder(tf.float32, [batchsize*self.encoder_len, inputdim])
        self.rec_inputs     = tf.placeholder(tf.float32, [batchsize*self.encoder_len, inputdim])
        self.pred_inputs    = tf.placeholder(tf.float32, [batchsize*(maxseq-self.encoder_len), inputdim])

        self.cellsize = rnnsize
        self._batchsize = batchsize
        self._maxseq = maxseq
        self._inputdim = inputdim
        self.numlayers = nlayers
        
        #self.early_stop = tf.placeholder(tf.int32, [self._batchsize])
        
        #self.cost_w = tf.placeholder(tf.float32, [self._batchsize,self._maxseq]) 
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.create_graph()
        
        
    def create_graph(self):    
        initializer = tf.random_uniform_initializer(-.1,.1) 

        """Define the network """    
        with tf.variable_scope('encoder'):
            self.encoder = self.createMultiLSTM(initializer)
        with tf.variable_scope('decoder_rec'):
            self.decoderRec = self.createMultiLSTM(initializer)
        with tf.variable_scope('decoder_pred'):
            self.decoderPred = self.createMultiLSTM(initializer)
            

        """set initial state"""
        self._initial_state = self.encoder.zero_state(self._batchsize, tf.float32)

        #if (self._inputdim == 64*64) or (self._inputdim == 48*48) or (self._inputdim == 32*32): 
        #    inputs = self.x
        #else:
        #    pred,conv_end,d1,d2,d3 = self._baseCNN.inference(
        #                    self.x, self._baseCNN.weights, self._baseCNN.biases, self.keep_prob)   
        #    if (self._inputdim == 400):
        #        inputs = d1
        #    elif self._inputdim == 128*64:
        #        inputs = conv_end

        #inputs = tf.nn.dropout(inputs, self.keep_prob)
        
        encoder_inputs = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
        rec_inputs = tf.nn.dropout(self.rec_inputs, self.keep_prob)
        pred_inputs = tf.nn.dropout(self.pred_inputs, self.keep_prob)
        
        Encoder_inputs = [tf.reshape(i, (self._batchsize, self._inputdim)) 
                          for i in tf.split(0, self.encoder_len, encoder_inputs)]
        Rec_inputs = [tf.reshape(i, (self._batchsize, self._inputdim)) for i in tf.split(0, self.encoder_len, rec_inputs)]
        Pred_inputs = [tf.reshape(i, (self._batchsize, self._inputdim)) 
                            for i in tf.split(0, self._maxseq-self.encoder_len , pred_inputs)]
        
        """inputs: list of (batchsize X inputdim)"""
        """retain the first \alpha for encoder input. \alpha = minseqlen/2"""
        """ and accordingly for the decoders"""   
            
        #inputs_enc = tf.slice(inputs,[0,0],[alpha,-1]) #retain the first alpha rows
            
        """TODO:    batch?
                    sequence_lengths for encoding-decoding

        """        
        with tf.variable_scope('encoder') as scope:
            self.encoder_outputs, hstate = tf.nn.rnn(self.encoder, Encoder_inputs, 
                                                     initial_state=self._initial_state,scope=scope)#, 
                                   #sequence_length=self.early_stop_enc)
        self.hstate = hstate                           
        with tf.variable_scope('decoder_rec') as scope:
            self.decoderRec_outputs, _ = tf.nn.rnn(self.decoderRec, Rec_inputs, initial_state=hstate,scope=scope)#, 
                                   #sequence_length=self.early_stop_decRec)
        with tf.variable_scope('decoder_pred') as scope:        
            self.decoderPred_outputs, _ = tf.nn.rnn(self.decoderPred, Pred_inputs, initial_state=hstate,scope=scope)#, 
                                   #sequence_length=self.early_stop_decPred)
        
            
    def calc_cost(self):
          
        self.labels_rec = tf.placeholder(tf.float32, [self._batchsize*self.encoder_len, self._inputdim])
          
        self.labels_pred = tf.placeholder(tf.float32, [self._batchsize*(self._maxseq - self.encoder_len),self._inputdim])  
  
        """outputs: batchsize(outer index) * maxseq(inner index) X inputdim"""

        output = self.decoderRec_outputs  
        output = tf.reshape(tf.concat(1, output), [-1, self.cellsize])        
        softmax_w = tf.get_variable("softmax_w_rec", [self.cellsize, self._inputdim])
        softmax_b = tf.get_variable("softmax_b_rec", [self._inputdim])
        self.output_rec = tf.matmul(output, softmax_w) + softmax_b
        
        output = self.decoderPred_outputs  
        output = tf.reshape(tf.concat(1, output), [-1, self.cellsize])
        softmax_w = tf.get_variable("softmax_w_pred", [self.cellsize, self._inputdim])
        softmax_b = tf.get_variable("softmax_b_pred", [self._inputdim])
        self.output_pred = tf.matmul(output, softmax_w) + softmax_b
        
        """error: squared difference"""
        self.loss = tf.add( tf.nn.l2_loss(tf.sub(self.labels_rec, self.output_rec )), 
                            tf.nn.l2_loss(tf.sub(self.labels_pred,self.output_pred)) )
                              
        return self.loss 
            
        
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
            preds,cost = sess.run( [self.predictions,self.cost], feed_dict={self.x: seq_xs, self.y: seq_ys, 
                            self.keep_prob: dropout, self.early_stop: seqlen, self.cost_w: w})      
        else:
            preds = sess.run( [self.predictions], feed_dict={self.x: seq_xs, #self.y: seq_ys, 
                            self.keep_prob: dropout, self.early_stop: seqlen, self.cost_w: w})      
        
        #Reshape to [batchsize X maxseq]
        preds = np.reshape(preds,[self._batchsize,self._maxseq])
        
        #sum per sequence
        sequence_prediction = np.divide(np.sum(w*preds, 1),np.sum(w,1))
        #print(w)
        #print(preds)
        #print(sequence_prediction)
        #print(seq_ys)

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
            corr_preds = seq_ys[:,0] == sequence_prediction

        return sequence_prediction, corr_preds, cost 
            
            
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
        
