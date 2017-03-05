import time

import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader
from imageReader import ImageReader

from myCNN import MyCNN

LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9_2_125/"

# Parameters
learning_rate = 0.0001
training_steps = 50000
display_step = 200
#display_step = 20

# Network Parameters
n_classes = 2 # classes 
maxseq = 16
batchsize = 1


testreader = SequenceReader("val/",istest=True,maxsequence=maxseq,isComposite=False)

#"""single-frame"""
#testreader = ImageReader("val/",istest=True)
#batchsize = 256

"""Construct and load pretrained CNN model"""

CNNmodel = MyCNN(n_classes,istrainable=True)
cost,batch_acc,preds = CNNmodel.calc_cost()

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

saverCNN = tf.train.Saver(tf.all_variables())
init1 = tf.initialize_all_variables()

sess = tf.InteractiveSession()

sess.run(init1)

#print(sess.run(CNNmodel.weights['out']))

ckpt = tf.train.get_checkpoint_state(LOAD_CNN_checkpoint_path)
if ckpt and ckpt.model_checkpoint_path:
    print(ckpt.model_checkpoint_path)
    print("[train_script]: LOADED CNN!")
    saverCNN.restore(sess, ckpt.model_checkpoint_path)
else:
    print("[train_script]: Failed to LOAD CNN!")
    raise SystemExit  
 
print("[train_script]: Testing")
#print(sess.run(CNNmodel.weights['out']))

step = 1
summ = 0
tmp_test_acc = []

testreader.index = 0
testreader._ignoresmall = 4
startTime = time.time()


predictions_file = open("logits_CNN.txt", "w+")

"""CNN works only with single sequences"""
while testreader._epochs < 1:
    seqlen_test, seq_xs_test, seq_ys_test = testreader.read_batch(batchsize)

    #seq_xs_test = seq_xs_test[:,:,2]

    """sequence level prediction"""
    logits,seq_prediction, corr_predictions, cost  = CNNmodel.sequence_predict(sess,seq_xs_test,1.,
                                                                        seqlen_test,seq_ys_test)
    
    predictions_file.write(""+str(logits[0])+","+str(logits[1])+"\n")

    #"""framewise prediction"""
    #seq_ys_test = seq_ys_test[0,0:seqlen_test[0]]
    #y = np.zeros([seqlen_test[0],2])
    #y[np.arange(0,seqlen_test[0]),np.array(seq_ys_test,dtype=np.int32)] = 1
    #seq_xs_test = seq_xs_test[0:seqlen_test[0],:]
    #
    #bacc = sess.run( [batch_acc], feed_dict={CNNmodel.x: seq_xs_test, CNNmodel.y: y, 
    #                        CNNmodel.keep_prob: 1.}) 
    #corr_predictions = bacc
    
    tmp_test_acc.extend(corr_predictions)
    if step % (display_step) == 0:
        print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_test_acc))))
        tmp_test_acc = []
        
    summ += np.mean(corr_predictions)
    step += 1


predictions_file.close()

print("Testing Accuracy:", "{:.5f}".format(summ/step))
print("Elapsed Time(s): ","{:1f}".format(time.time()-startTime))

