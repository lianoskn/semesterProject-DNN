
import tensorflow as tf

class TensorNet:
    
    def __init__(self,nclasses,istrainable=True,name=''):
        self._nclasses = nclasses
        #with tf.variable_scope(name):
        self.x = tf.placeholder(tf.float32, [None,4096])
        self.y = tf.placeholder(tf.int32, [None,2])
        self.keep_prob = tf.placeholder(tf.float32)
        self._istrainable = istrainable
        self._scopename = name
    def predict(self):
        True
    
    def train(self):
        True
            