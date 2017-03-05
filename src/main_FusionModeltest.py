
import os
import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader

from myCNN import MyCNN
from myFusionRNN import MyFusionRNN
from myFusionCNN import MyFusionCNN

LOAD_CNN = True
LOAD_RNN = True

LOAD_CNNflow_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_flowCNNnamed/"
LOAD_CNNimg_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgCNNcomposite/"

#rnn_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgflowRNNd1_2x64/"
rnn_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_fuseCNN_fc62layers_end2end/"

train_err_file = open("train_err_imgflowRNNd1_2x64.txt", "a+")
# Parameters
learning_rate = .0001
training_steps = 200000
display_step = 100

# Network Parameters
n_classes = 2 # classes 
dropout = .5 #0.70 # Dropout, probability to keep units
nlayers = 2
rnnsize = 64
maxseq = 16
batchsize = 16

dirpath = "/home/lianos91/Desktop/training_patches/train_patches_128_64_rich/"
datapath = "val/"
testreader = SequenceReader(dirpath,datapath,istest=True,maxsequence=16,isComposite=True)

"""Construct and load pretrained CNN model"""

with tf.variable_scope('flow'):
    CNNflowbase = MyCNN(n_classes,istrainable=False)

sess = tf.InteractiveSession()
saverCNN = tf.train.Saver(tf.all_variables())

init1 = tf.initialize_all_variables()
sess.run(init1)

existing = set(tf.all_variables())

if LOAD_CNN:
    ckpt = tf.train.get_checkpoint_state(LOAD_CNNflow_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED CNN!")
        saverCNN.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: Failed to LOAD CNN!")
        raise SystemExit  
    
CNNbase = MyCNN(n_classes,istrainable=False)

saverCNN2 = tf.train.Saver(set(tf.all_variables()) - existing)

init1 = tf.initialize_variables(set(tf.all_variables()) - existing)
sess.run(init1)

if LOAD_CNN:
    ckpt = tf.train.get_checkpoint_state(LOAD_CNNimg_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED CNN!")
        saverCNN2.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: Failed to LOAD CNN!")
        raise SystemExit

"""Construct RNN"""
#rnn_net = MyFusionRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,CNNflowbase,2*400,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,128*64,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,64*64,nlayers)

rnn_net = MyFusionCNN(n_classes,CNNbase,CNNflowbase,2*400,True)
batchsize  = 1

# Define loss 
#cost,preds = rnn_net.calc_cost()
cost,batch_accuracy,_ = rnn_net.calc_cost()

"""Define Optimization settings"""

max_grad_norm = 4
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
        
##get variables e
temp = set(tf.all_variables())

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

print("[train_script]: RNN constructed & Optimization is set")

#remaining variables = ADAM variables + trainable variables
remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
init2 = tf.initialize_variables(remaining_vars)

saverRNN = tf.train.Saver(tf.all_variables())

sess.run(init2)

print("[train_script]: Initialized")
   
ckpt = tf.train.get_checkpoint_state(rnn_checkpoint_path)
print(ckpt.model_checkpoint_path)

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

step = 1
summ = 0
tmp_test_acc = []

testreader.index = 0
testreader._ignoresmall = 4

predictions_file = open("logits_composite_imgflow_fusedCNN2layersd1_end2end.txt", "w+")

while testreader._epochs < 1:

    seqlen_test,seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)
    
    # 2:image patch
    # 1: optical flow
    seq_xs_test = [seq_xs_test[:,:,2],seq_xs_test[:,:,1]]
         
    #w = rnn_net.weighting(seqlen_test,weights='linear')
    #pr,corr_predictions,cost,logits = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)
    #for i in range(0,len(pr)):
    #    predictions_file.write(""+str(logits[0,i])+","+str(logits[1,i])+"\n")
     
    logits,seq_prediction, corr_predictions, cost  = rnn_net.sequence_predict(sess,seq_xs_test,1.,
                            seqlen_test,seq_ys_test)
    
    predictions_file.write(""+str(logits[0])+","+str(logits[1])+"\n")
 
    tmp_test_acc.extend(corr_predictions)
    if step % (display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_test_acc))))
        tmp_test_acc = []
        
    summ += np.mean(corr_predictions)
    step += 1

predictions_file.close()
print("Testing Accuracy:", "{:.5f}".format(summ/step))
