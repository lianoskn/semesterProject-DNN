
import os
import numpy as np

import tensorflow as tf

#from sequenceReader import SequenceReader
#from compositeSequenceReader import CompositeSequenceReader
#from compositeReader import CompositeReader
from imageReader import ImageReader

from myCNN import MyCNN
from myFusionRNN import MyFusionRNN
from myFusionCNN import MyFusionCNN

LOAD_CNN = True
LOAD_FUSED = True
SAVE_MODEL = True
baseTrainable = True
#checkfile = "fuseCNN1laylr0d0001_d1.ckpt"
checkfile = "fuseCNN2layerslr0d0001_d1.ckpt"

LOAD_CNNflow_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_flowCNNnamed/"
LOAD_CNNimg_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgCNNcomposite/"

dirpath = "/home/lianos91/Desktop/training_patches/train_patches_128_64_rich/"
datapath = "train/"
#save_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgflowRNNd1_2x64/"
#save_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_fuseCNN_fc61layer_end2end/"
save_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_fuseCNN_fc62layers_end2end/"

train_err_file = open("train_err_imgflowRNNd1_2x64.txt", "a+")

# Parameters
learning_rate = .0001
training_steps = 200000
display_step = 50

# Network Parameters
n_classes = 2 # classes 
dropout = .5 #0.70 # Dropout, probability to keep units
nlayers = 2
rnnsize = 64
maxseq = 16
batchsize = 16

#reader = CompositeSequenceReader("train/",istest=False,maxsequence=maxseq)
reader = ImageReader(dirpath,datapath,istest=False,isComposite=True)
#reader = ImageReader(dirpath,datapath,istest=False,isComposite=True)

"""Construct and load pretrained CNN model"""

with tf.variable_scope('flow'):
    CNNflowbase = MyCNN(n_classes,istrainable=baseTrainable)

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
    
CNNbase = MyCNN(n_classes,istrainable=baseTrainable)

saverCNN2 = tf.train.Saver(set(tf.all_variables()) - existing)

init1 = tf.initialize_variables(set(tf.all_variables()) - existing)
sess.run(init1)

existing = set(tf.all_variables())

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

CNNmodel = MyFusionCNN(n_classes,CNNbase,CNNflowbase,2*400,True)
batchsize  = 128

fusion_model_vars = set(tf.all_variables()) - existing

# Define loss 
#cost,preds = rnn_net.calc_cost()
cost,batch_accuracy,_ = CNNmodel.calc_cost()

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
remaining_vars = (set(tf.all_variables()) - temp ) | fusion_model_vars# | set(tf.trainable_variables())
#remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
init2 = tf.initialize_variables(remaining_vars)

saverRNN = tf.train.Saver(tf.all_variables())

sess.run(init2)

print("[train_script]: Initialized")

step = 1
   
if LOAD_FUSED:
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

print("[train_script]: Start training")
   
# Keep training until reach max iterations
tmp_train_acc = []
tmp_test_acc = []
train_cost = []

reader._ignoresmall = 5

while step < training_steps:
    
    if (step == -1):
        reader._ignoresmall = 0

    #seqlen,seq_xs, seq_ys = reader.read_batch(batchsize)
    seq_xs, seq_ys = reader.read_batch(batchsize)

    #optical flow:  1
    #patch:         2
    seq_xs = np.vstack([seq_xs[:,:,2],seq_xs[:,:,1]])
     
            
    """seq_xs: maxseq(outer) * batchsize(inner) X img_dim """
    """seq_ys: batchsize X maxseq """
    """w     : batchsize X maxseq """
    """seqlen: batchsize X 1      """
    #w = rnn_net.weighting(seqlen,[],'uniform')
    # Fit training using sequence data
    #sess.run( [train_op], feed_dict={rnn_net.x: seq_xs,
    #                                 rnn_net.y: seq_ys, 
    #                                 rnn_net.keep_prob: dropout, rnn_net.early_stop: seqlen, 
    #                                 rnn_net.cost_w: w})  
    
    sess.run( [train_op], feed_dict={CNNmodel.x: seq_xs, CNNmodel.y: seq_ys, 
                            CNNmodel.keep_prob: dropout})  
    
    
    """"""
    """get training and test error"""
    if step % 2 == 0:
        
        
        corr_predictions = sess.run( [batch_accuracy], feed_dict={CNNmodel.x: seq_xs, CNNmodel.y: seq_ys, 
                            CNNmodel.keep_prob: 1.}) 
        
        #pr,corr_predictions,cst,_ = rnn_net.predict(sess,seq_xs,1.,seqlen,w,seq_ys)   
        tmp_train_acc.extend(corr_predictions)
        #train_cost.append(cst)

    if step % display_step == 0:
        #output mean training accuracy of past sequences
        m = np.mean(np.array(tmp_train_acc,np.float32))
        #m3 = np.mean(np.array(train_cost,np.float32))
        #m4 = np.std(np.array(train_cost,np.float32))
        print("[train_script]: Step: "+str(step)+ ", train_acc {:.3f}".format(m))
        #print("train_mean_cost {:.3f}".format(m3)+ ", train_cost_std {:.3f}".format(m4) )
        #train_err_file.write(""+str(m3)+","+str(m)+"\n")
        tmp_train_acc = []
        train_cost = []

    #checkpoint save
    if step % (5*display_step) == 0 and SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
        saverRNN.save(sess, checkpoint_path, global_step=step)        
    step += 1
    
    
train_err_file.close()

reader.terminate = True
print("[train_script]: Optimization Finished!")
if SAVE_MODEL:
    print("[train_script]: checkpoint")
    checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
    saverRNN.save(sess, checkpoint_path, global_step=step)


