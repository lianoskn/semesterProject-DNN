
import os
import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader
from myCNN import MyCNN
from myRNN import MyRNN

LOAD_CNN = False
LOAD_RNN = False
SAVE_MODEL = True

checkfile = "RNNlr0d0001_d1_b16_1x64_noPreT_dr55_end2end.ckpt"

LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9_2_125/"
save_checkpoint_path = "/home/lianos91/Desktop/training_patches/rnn_d1_1x64_noPreT_dr55_end2end"

train_err_file = open("train_err_d1_b16_1x64_noPreT_dr55_end2end.txt", "a+")
# Parameters
learning_rate = .0001
training_steps = 200000
display_step = 30

# Network Parameters
n_classes = 2 # classes 
dropout = .55 #0.70 # Dropout, probability to keep units
nlayers = 1
rnnsize = 64
maxseq = 16
batchsize = 16

trainreader = SequenceReader("train_largerThan4/",istest=False,maxsequence=maxseq)
#trainreader = SequenceReader("train_tiny/",istest=False,maxsequence=maxseq)
#trainreader = MySequenceReader("train/",istest=False,maxsequence=maxseq)

#testreader = MySequenceReader("val/",istest=True,maxsequence=maxseq)


"""Construct and load pretrained CNN model"""

CNNbase = MyCNN(n_classes,istrainable=True)

sess = tf.InteractiveSession()
saverCNN = tf.train.Saver(tf.all_variables())

init1 = tf.initialize_all_variables()
sess.run(init1)

if LOAD_CNN:
    ckpt = tf.train.get_checkpoint_state(LOAD_CNN_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED CNN!")
        saverCNN.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: Failed to LOAD CNN!")
        raise SystemExit  


"""Construct RNN"""
rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,400,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,128*64,nlayers)
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,64*64,nlayers)

# Define loss 
cost,preds = rnn_net.calc_cost()

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

##remaining variables = ADAM variables + trainable variables
remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
init2 = tf.initialize_variables(remaining_vars)
#init2 = set(tf.all_variables())

# Build the summary operation based on the TF collection of Summaries.
#summary_op = tf.merge_all_summaries()
#summaries
#cost_summ = tf.scalar_summary("cost", cost)

saverRNN = tf.train.Saver(tf.all_variables())

#merged = tf.merge_all_summaries()
#summ_writer = tf.train.SummaryWriter(trainreader.dirpath)
sess.run(init2)

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

print("[train_script]: Start training")
   
# Keep training until reach max iterations
tmp_train_acc = []
tmp_test_acc = []
train_cost = []

trainreader._ignoresmall = 5

while step < training_steps:
    
    if (step == -1):
        trainreader._ignoresmall = 0

    seqlen,seq_xs, seq_ys = trainreader.read_batch(batchsize)

    w = rnn_net.weighting(seqlen,[],'uniform')
            
    """seq_xs: maxseq(outer) * batchsize(inner) X img_dim """
    """seq_ys: batchsize X maxseq """
    """w     : batchsize X maxseq """
    """seqlen: batchsize X 1      """
    
    # Fit training using sequence data
    sess.run( [train_op], feed_dict={rnn_net.x: seq_xs, rnn_net.y: seq_ys, 
                                   rnn_net.keep_prob: dropout, rnn_net.early_stop: seqlen, 
                                   rnn_net.cost_w: w})  
    """"""
    """get training and test error"""
    if step % 2 == 0:
        
        pr,corr_predictions,cst = rnn_net.predict(sess,seq_xs,1.,seqlen,w,seq_ys)   
        tmp_train_acc.extend(corr_predictions)
        train_cost.append(cst)

    if step % display_step == 0:
        #output mean training accuracy of past sequences
        m = np.mean(np.array(tmp_train_acc,np.float32))
        m3 = np.mean(np.array(train_cost,np.float32))
        m4 = np.std(np.array(train_cost,np.float32))
        print("[train_script]: Step: "+str(step)+ ", train_acc {:.3f}".format(m))
        print("train_mean_cost {:.3f}".format(m3)+ ", train_cost_std {:.3f}".format(m4) )
        train_err_file.write(""+str(m3)+","+str(m)+"\n")

        tmp_train_acc = []
        train_cost = []

        #write summary
        #summ_writer.add_summary(m, step)
           
    #checkpoint save
    if step % (20*display_step) == 0 and SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
        saverRNN.save(sess, checkpoint_path, global_step=step)        
    step += 1
    
    
train_err_file.close()

trainreader.terminate = True
print("[train_script]: Optimization Finished!")
if SAVE_MODEL:
    print("[train_script]: checkpoint")
    checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
    saverRNN.save(sess, checkpoint_path, global_step=step)

#acquire test set

print("[train_script]: Testing")

step=0
summ = 0

testreader = SequenceReader("val/",istest=True,maxsequence=maxseq)

testreader.index = 0

tmp_train_acc = []

while testreader._epochs < 1:

    seqlen_test,seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)
        
    w = rnn_net.weighting(seqlen_test)
    pr,corr_predictions,_ = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)
    
    tmp_train_acc.extend(corr_predictions)
    if step % (2*display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_train_acc))))
        tmp_train_acc = []
    summ += np.mean(corr_predictions)
    step += 1

print("Testing Accuracy:", "{:.5f}".format(summ/step))

