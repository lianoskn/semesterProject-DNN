import os
import numpy as np
import tensorflow as tf
from mySequenceReader import MySequenceReader
from encoderDecoderRNN import EncoderDecoderRNN

LOAD_MODEL = True

checkfile = "enc_2048_i32_lrd00005.ckpt"
save_checkpoint_path = "/home/lianos91/Desktop/training_patches/EncoderRNN_2048_i32_lrd00005/"

""" Learning Parameters"""
learning_rate = .00005
training_steps = 300000
display_step = 1000

# Network Parameters
n_classes = 2 # classes 
dropout = .7 #0.70 # Dropout, probability to keep units
nlayers = 1
rnnsize = 2048
maxseq = 16
batchsize = 1
encoder_prcnt = 2./3.
encoder_length = round(encoder_prcnt*maxseq)

img_dim = 32

#testreader = MySequenceReader("train_largerThan4/",istest=False,maxsequence=maxseq)
#testreader = MySequenceReader("train_tiny/",istest=False,maxsequence=maxseq,imdim=img_dim)

testreader = MySequenceReader("train_16/",istest=True,maxsequence=maxseq,imdim=img_dim)
testreader._ignoresmall = 1

encoder = EncoderDecoderRNN(n_classes,rnnsize,batchsize,maxseq,img_dim*img_dim,nlayers,encoder_length)
#  __init__(self, nclasses,rnnsize,batchsize,maxseq,inputdim,nlayers = 1):

cost = encoder.calc_cost()

"""Define Optimization settings"""
max_grad_norm = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
temp = set(tf.all_variables())

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

init = tf.initialize_all_variables()

sess = tf.InteractiveSession()

saver = tf.train.Saver(tf.all_variables())

sess.run(init)

step = 0

if LOAD_MODEL:   
    ckpt = tf.train.get_checkpoint_state(save_checkpoint_path)
    print(ckpt.model_checkpoint_path)
    
    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED RNN")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: failed to load")
        raise SystemExit
    last_checkpoint_path = ckpt.model_checkpoint_path
    step = 1+int(last_checkpoint_path[last_checkpoint_path.rindex('-')+1:])
    
print(step)




z = np.zeros([1,img_dim*img_dim])

tmpcost = []

Data = np.array()

while testreader._epochs < 1:
    seqlen,seq_xs, seq_ys = testreader.read_batch(batchsize)
    seq_xs = seq_xs/255.    
    
    enc_input = seq_xs[0:encoder_length,:]
    dec_rec_input = np.vstack((z,(enc_input[::-1,:])[0:-1]))
    dec_pred_input = np.vstack((z,seq_xs[encoder_length:-1,:]))
    
    rec_label = enc_input[::-1,:]
    pred_label = seq_xs[encoder_length::,:]
    # Fit training using sequence data
    enc_state,_,_ = sess.run( [encoder.hstate, encoder.output_rec, encoder.output_pred], 
                                feed_dict={encoder.encoder_inputs: enc_input,
                                    encoder.rec_inputs:  dec_rec_input,
                                    encoder.pred_inputs: dec_pred_input,
                                    encoder.labels_rec:  rec_label,
                                    encoder.labels_pred: np.float32(pred_label),
                                    encoder.keep_prob:   dropout}) 
    Data.                     
                         
                         
                         
