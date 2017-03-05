import time

import numpy as np

import tensorflow as tf

from myRNNreader import MyRNNreader
from myCNN import MyCNN
from myRNN import MyRNN

from myReader import MyReader

LOAD_RNN = True

checkfile = "myRNNlr0d0005_conv_b32_2x64_no5.ckpt"

LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9/"
rnn_checkpoint_path = "/home/lianos91/Desktop/training_patches/model15d1_2nd(bkp)/"

# Parameters
learning_rate = 0.007
training_steps = 50000
display_step = 20

# Network Parameters
n_classes = 2 # classes 
dropout = .8#0.70 # Dropout, probability to keep units
nlayers = 2
rnnsize = 64
maxseq = 16
batchsize = 32

testreader = MyRNNreader("val/",istest=True,maxsequence=maxseq)

"""Construct and load pretrained CNN model"""

CNNbase = MyCNN(n_classes,istrainable=False)

sess = tf.InteractiveSession()

#saverCNN = tf.train.Saver(tf.all_variables())

#init1 = tf.initialize_all_variables()
#sess.run(init1)

#ckpt = tf.train.get_checkpoint_state(LOAD_CNN_checkpoint_path)
#if ckpt and ckpt.model_checkpoint_path:
#    print("[train_script]: LOADED CNN!")
#    saverCNN.restore(sess, ckpt.model_checkpoint_path)
#else:
#    print("[train_script]: Failed to LOAD CNN!")
#    raise SystemExit  



"""Construct RNN"""
rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,400,nlayers)
# Define loss 
cost,preds = rnn_net.calc_cost()

"""Define Optimization settings"""

max_grad_norm = 5
        
#_lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
        
#optimizer = tf.train.GradientDescentOptimizer(_lr)

##get variables e
temp = set(tf.all_variables())

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

print("[train_script]: RNN constructed & Optimization is set")

#remaining variables = ADAM variables + trainable variables
#remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
#init2 = tf.initialize_variables(remaining_vars)

#if load RNN only:
init2 = tf.initialize_all_variables()

# Build the summary operation based on the TF collection of Summaries.
#summary_op = tf.merge_all_summaries()
#summaries
#cost_summ = tf.scalar_summary("cost", cost)

saverRNN = tf.train.Saver(tf.all_variables())

#merged = tf.merge_all_summaries()
#summ_writer = tf.train.SummaryWriter(trainreader.dirpath)
sess.run(init2)

print("[train_script]: Initialized")

step = 0
   
if LOAD_RNN:
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

print("[train_script]: Start training")

tmp_test_acc = []

print("[train_script]: Testing")

step = 0
summ = 0

testreader.index = 0
testreader._ignoresmall = 1

startTime = time.time()

while testreader._epochs < 1:
    #seqlen,seq_xs, seq_ys = testreader.read_sequence()
    seqlen_test,seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)
    
    #w=np.zeros([batchsize,maxseq])
    #for i in range(0,batchsize):
    #    denom = np.sum(np.array([k for k in range(0,seqlen_test[i])]))
    #    for j in range(0,maxseq):
    #        w[i,j] = 1.*(j<seqlen_test[i])
            #w[i,j] = ((j+1)/denom)*seqlen[i]*(j<seqlen[i])
    
    ##print(seqlen_test)
    ##print(w)
     
    #preds = sess.run( [predictions], feed_dict={rnn_net.x: seq_xs_test, rnn_net.y: seq_ys_test, 
    #                    rnn_net.keep_prob: 1., rnn_net.early_stop: seqlen_test, rnn_net.cost_w: w})          
    w = rnn_net.weighting(seqlen_test,[],'linear')
    pr,corr_predictions,_ = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)
    #Reshape to [batchsize X maxseq]
    #preds = np.reshape(preds,[batchsize,maxseq])
    
    #sum per sequence
    #pr = np.sum(w*preds, 1)
    
    #find the sequences with equal votes (ambiguous)
    #iseq = pr == np.array(seqlen_test)/2 
    #set the prediction label for not ambiguous sequences
    #pr = pr > np.array(seqlen_test)/2.          

    # for the ambiguous, label the sequence according to the label of the last element
    #if np.sum(iseq) > 0:
    #    preds = preds[iseq,:]
    #    seqlen_test = np.array(seqlen_test)
    #    pr[iseq] = preds[:,seqlen_test[iseq]-1];
    
    #prediction correctness
    #corr_predictions = seq_ys_test[:,0] == pr
    tmp_test_acc.extend(corr_predictions)
    if step % (display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_test_acc))))
        tmp_test_acc = []
        
    summ += np.mean((corr_predictions))
    step += 1

print("Testing Accuracy:", "{:.5f}".format(summ/step))

print("Elapsed Time(s): ","{:1f}".format(time.time()-startTime))

