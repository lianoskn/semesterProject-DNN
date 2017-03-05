import time

import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader
from compositeSequenceReader import CompositeSequenceReader

from myCNN import MyCNN
from myRNN import MyRNN


LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_flowCNNnamed/"
rnn_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_flowRNNcomposite_d1_2x128/"
#LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgCNNcomposite/"
#rnn_checkpoint_path = "/home/lianos91/Desktop/training_patches/model_imgRNNcomposite_d1_2x128/"

# Parameters
learning_rate = 0.0005
training_steps = 4000
display_step = 20

# Network Parameters
n_classes = 2 # classes 
dropout = .8#0.70 # Dropout, probability to keep units
nlayers = 2
rnnsize = 128
maxseq = 16
batchsize = 32

#testreader = SequenceReader("val/",istest=True,maxsequence=maxseq)
testreader = CompositeSequenceReader("val/",istest=True,maxsequence=maxseq)

"""Construct and load pretrained CNN model"""
with tf.variable_scope('flow'):
    CNNbase = MyCNN(n_classes,istrainable=False)

sess = tf.InteractiveSession()

saverCNN = tf.train.Saver(tf.all_variables())

init1 = tf.initialize_all_variables()
sess.run(init1)

ckpt = tf.train.get_checkpoint_state(LOAD_CNN_checkpoint_path)
if ckpt and ckpt.model_checkpoint_path:
    print("[train_script]: LOADED CNN!")
    saverCNN.restore(sess, ckpt.model_checkpoint_path)
else:
    print("[train_script]: Failed to LOAD CNN!")
    raise SystemExit  


"""Construct RNN"""
#rnn_net = MyRNN(n_classes,rnnsize,batchsize,maxseq,CNNbase,128*64,nlayers)
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

##remaining variables = ADAM variables + trainable variables
remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
init2 = tf.initialize_variables(remaining_vars)

##if load RNN only:
#init2 = tf.initialize_all_variables()

# Build the summary operation based on the TF collection of Summaries.
#summary_op = tf.merge_all_summaries()
#summaries
#cost_summ = tf.scalar_summary("cost", cost)

saverRNN = tf.train.Saver(tf.all_variables())

#merged = tf.merge_all_summaries()
#summ_writer = tf.train.SummaryWriter(trainreader.dirpath)
sess.run(init2)

print("[train_script]: Initialized. Loading...")

step = 0
   
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

tmp_test_acc = []
v_acc = []

print("[train_script]: Testing")

step = 1
summ = 0

testreader.index = 0
testreader._ignoresmall = 4

startTime = time.time()

predictions_file = open("logits_compositeflowRNN.txt", "w+")

while testreader._epochs < 1:

    seqlen_test,seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)
  
    #optical flow:  1
    #patch:         2
    seq_xs_test = seq_xs_test[:,:,1]

    w = rnn_net.weighting(seqlen_test,[],'linear')
    pr,corr_predictions,_,logits = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)

    for i in range(0,len(pr)):
        predictions_file.write(""+str(logits[0,i])+","+str(logits[1,i])+"\n")
           
    tmp_test_acc.extend(corr_predictions)
    if step % (display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_test_acc))))
        v_acc.append(np.mean(np.mean(tmp_test_acc)))
        tmp_test_acc = []
    summ += np.mean((corr_predictions))
    step += 1


predictions_file.close()

print("Testing Accuracy:", "{:.5f}".format(summ/step))
print("Testing Variance:", "{:.5f}".format(np.sum(np.power(np.array(v_acc) - (summ/step),2))))
print("Elapsed Time(s): ","{:1f}".format(time.time()-startTime))

