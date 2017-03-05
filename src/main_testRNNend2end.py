
import os
import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader
from myCNN import MyCNN
from myRNN import MyRNN

LOAD_RNN = True

checkfile = "RNNlr0d0001_d1_b16_128_end2end.ckpt"

LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9_2_125/"
save_checkpoint_path = "/home/lianos91/Desktop/training_patches/rnn_d1_end2end/"


train_err_file = open("train_err_d1_2x64_end2end.txt", "a+")
# Parameters
learning_rate = .0001
training_steps = 200000
display_step = 50

# Network Parameters
n_classes = 2 # classes 
dropout = .7 #0.70 # Dropout, probability to keep units
nlayers = 2
rnnsize = 64
maxseq = 16
batchsize = 16

"""Construct and load pretrained CNN model"""

CNNbase = MyCNN(n_classes,istrainable=True)

"""Construct RNN"""
rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,400,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,128*64,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,64*64,nlayers)

# Define loss 
cost,preds = rnn_net.calc_cost()

sess = tf.InteractiveSession()

"""Define Optimization settings"""

max_grad_norm = 4
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
      
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

print("[train_script]: RNN constructed & Optimization is set")

sess.run(tf.initialize_all_variables())

saverRNN = tf.train.Saver(tf.all_variables())
print("[train_script]: Initialized")

step = 1
   
if LOAD_RNN:
    ckpt = tf.train.get_checkpoint_state(save_checkpoint_path)
    print(ckpt.model_checkpoint_path)    
    print(save_checkpoint_path)

    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED RNN")
        saverRNN.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: failed to load")
        raise SystemExit
    last_checkpoint_path = ckpt.model_checkpoint_path
    step = 1+int(last_checkpoint_path[last_checkpoint_path.rindex('-')+1:])
    print(step)

print("[train_script]: Testing")

step=0
summ=0

testreader = SequenceReader("val/",istest=True,maxsequence=maxseq)
testreader._ignoresmall = 5

testreader.index = 0

tmp_train_acc = []

while testreader._epochs < 1:

    seqlen_test,seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)
        
    w = rnn_net.weighting(seqlen_test,[],'uniform')
    pr,corr_predictions,_ = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)
    
    tmp_train_acc.extend(corr_predictions)
    if step % (2*display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_train_acc))))
        tmp_train_acc = []
    summ += np.mean(corr_predictions)
    step += 1

print("Testing Accuracy:", "{:.5f}".format(summ/step))

