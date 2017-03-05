
import tensorflow as tf
from tensorNet import TensorNet
import numpy as np

class MyCNN(TensorNet):
    
    def __init__(self,nclasses,istrainable=True,name = ''):
        super().__init__( nclasses,istrainable,name)
        self.initializevars()
        self.y = tf.placeholder(tf.float32, [None,2])

        """predictions for individual frames"""
        self.logits,_,_,_,_ = self.inference(self.x, self.weights, self.biases, self.keep_prob)
        logits = self.logits
        self.predictions = tf.argmax(logits,1)

        #"""prediction for a single sequence"""
        #self.sequence_prediction = tf.greater_equal(tf.reduce_mean(self.predictions),
        #                                           tf.constant(tf.shape(self.predictions)[0]))
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))

    def calc_cost(self):
        """evaluate accuracy on indiv. frames"""
        
        correct_pred = tf.equal(self.predictions, tf.argmax(self.y,1))
        batchaccuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))    
        return self.cost, batchaccuracy, self.predictions
    
    """prediction for a single sequence"""    
    def sequence_predict(self,sess,seq_xs,dropout,seqlen,seq_ys = []):

        
        """must transform seq_ys to one-hot, IF USE TF FOR CORRECTION LABELLING"""
        if len(seq_ys) > 0:
            # since we have 1 sequence
            seq_ys = seq_ys[0,0:seqlen[0]]
            y = np.zeros([seqlen[0],2])
            y[np.arange(0,seqlen[0]),np.array(seq_ys,dtype=np.int32)] = 1

        """cut spare entries of xs (added by the reader)"""
        seq_xs = seq_xs[0:seqlen[0],:]
        
        cost = -1
        if len(seq_ys) > 0:
            llogits, predictions,cost = sess.run( [self.logits, self.predictions,self.cost], feed_dict={self.x: seq_xs, self.y: y, 
                            self.keep_prob: dropout})      
        else:
            llogits,predictions = sess.run( [self.logits, self.predictions], feed_dict={self.x: seq_xs, 
                            self.keep_prob: dropout})      
       
      
        seq_prediction = np.sum(predictions) >= seqlen[0]/2.
               
        #if seq_ys is provided, then output also correct predictions
        corr_preds = []
        if len(seq_ys) > 0:
            corr_preds = (seq_ys[0] == seq_prediction)

        return np.sum(llogits,0),seq_prediction, [corr_preds], cost 
    
    
    def seq_acc(self):
        pred,_,_,_ = self.inference(self.x, self.weights, self.biases, self.keep_prob)
        pred = tf.argmax(pred,1)
        
        sequence_prediction = tf.greater_equal(tf.reduce_mean(pred),tf.constant(tf.size(pred)[0]))
        correct_preds = tf.equal(pred, tf.argmax(self.y,1))
        acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))    
        res = tf.greater_equal(acc,tf.div(tf.cast(tf.size(acc), tf.float32),tf.constant(2.0)))
        return res 
        
    def conv2d(self,name, l_input, w, b,s=1):
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, s, s, 1], padding='SAME'),b), name=name)
         
    def max_pool(self,name, l_input, k):
            return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
         
    def norm(self,name, l_input, lsize=4):
            return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        
    def inference(self,_X, _weights, _biases, _dropout):
        # Reshape input picture
        _X = tf.reshape(_X, shape=[-1, 64, 64, 1])
    
        # Convolution Layer
        conv1 = self.conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
    
        # Convolution Layer
        conv2 = self.conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = self.norm('norm2', pool2, lsize=4)
    
        # Convolution Layer
        conv3 = self.conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
       
        conv4 = self.conv2d('conv4', conv3, _weights['wc4'], _biases['bc4'])

        conv5 = self.conv2d('conv3', conv4, _weights['wc5'], _biases['bc5']) 
        pool5 = self.max_pool('pool5', conv5, k=2)

        # Fully connected layer
        dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape pool5 output to fit dense layer input
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
        dense1 = tf.nn.dropout(dense1,_dropout)
        
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
        dense2 = tf.nn.dropout(dense2,_dropout)
        
        dense3 = tf.nn.relu(tf.matmul(dense2, _weights['wd3']) + _biases['bd3'], name='fc3') # Relu activation
        dense3 = tf.nn.dropout(dense3,_dropout)
        
        # Output, class logits
        out = tf.matmul(dense3, _weights['out']) + _biases['out']
        
        self.out = out
        self.dense1 = dense1
        self.conv5 = conv5
        
        return out,pool5,dense1,dense2,dense3
    
    # Store layers weight & bias
    def initializevars(self):
            self.weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128],stddev=0.01),trainable = self._istrainable),
                'wc2': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01),trainable = self._istrainable),
                'wc3': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01),trainable = self._istrainable),
                'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01),trainable = self._istrainable),
                'wc5': tf.Variable(tf.random_normal([3, 3, 128, 128],stddev=0.01),trainable = self._istrainable),
                'wd1': tf.Variable(tf.random_normal([np.int(np.ceil(64*64/(8*8))*128), 400],stddev=0.01),trainable = self._istrainable),
                'wd2': tf.Variable(tf.random_normal([400, 400],stddev=0.01),trainable = self._istrainable),
                'wd3': tf.Variable(tf.random_normal([400, 200],stddev=0.01),trainable = self._istrainable),
                'out': tf.Variable(tf.random_normal([200, self._nclasses],stddev=0.01),trainable = self._istrainable)
            }
            self.biases = {
                'bc1': tf.Variable(tf.zeros([128]),trainable = self._istrainable),
                'bc2': tf.Variable(tf.zeros([128]),trainable = self._istrainable),
                'bc3': tf.Variable(tf.zeros([128]),trainable = self._istrainable),
                'bc4': tf.Variable(tf.zeros([128]),trainable = self._istrainable),
                'bc5': tf.Variable(tf.zeros([128]),trainable = self._istrainable),
                'bd1': tf.Variable(tf.zeros([400]),trainable = self._istrainable),
                'bd2': tf.Variable(tf.zeros([400]),trainable = self._istrainable),
                'bd3': tf.Variable(tf.zeros([200]),trainable = self._istrainable),
                'out': tf.Variable(tf.zeros([self._nclasses]),trainable = self._istrainable)
            }     
          
def CNN_sequence_predict(cnn,sess,seq_xs,dropout,seqlen,seq_ys = []):
 
        """must transform seq_ys to one-hot, IF USE TF FOR CORRECTION LABELLING"""
        if len(seq_ys) > 0:
            y = np.zeros([seqlen[0],2])
            y[np.arange(0,seqlen[0]),np.array(seq_ys,dtype=np.int32)] = 1
 
        cost = -1
        if len(seq_ys) > 0:
            llogits, predictions,cost = sess.run( [cnn.logits, cnn.predictions,cnn.cost], feed_dict={cnn.x: seq_xs, cnn.y: y, 
                            cnn.keep_prob: dropout})      
        else:
            llogits,predictions = sess.run( [cnn.logits, cnn.predictions], feed_dict={cnn.x: seq_xs, 
                            cnn.keep_prob: dropout})    
        
        seq_prediction = np.sum(predictions) >= seqlen[0]/2.
        
        #if seq_ys is provided, then output also correct predictions
        corr_preds = []
        if len(seq_ys) > 0:
            corr_preds = (seq_ys[0] == seq_prediction)

        return np.sum(llogits,0),seq_prediction, [corr_preds], cost 
