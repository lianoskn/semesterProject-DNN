
import os
import sys

import numpy as np

import tensorflow as tf

from sequenceReader import SequenceReader
from myCNN import MyCNN
from myRNN import MyRNN
from configuration import Config
from imageReader import ImageReader


conf = Config()
conf.parseConfigFile(sys.argv[1]) 

train_err_file = open(conf.err_plot, "a+")
# Parameters

datafolder  = (conf.testdata, conf.traindata)[conf.isTest == 1]
if conf.IS_CNN and not conf.isTest:
    reader = ImageReader(conf.root_dir,datafolder,istest=conf.isTest)
else:
    reader = SequenceReader(conf.root_dir,datafolder,istest=conf.isTest,maxsequence=conf.maxseq)

    
models = []
"""Construct and load pretrained CNN model"""
if conf.IS_CNN:
    CNNbase = MyCNN(conf.n_classes, conf.is_cnn_trainable)
    models.append(CNNbase)

sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

init1 = tf.initialize_all_variables()
sess.run(init1)

if conf.LOAD_CNN:
    ckpt = tf.train.get_checkpoint_state(conf.LOAD_CNN_checkpoint_dir)
    print(ckpt.model_checkpoint_path)    
    print(conf.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED CNN!")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: Failed to LOAD CNN!")
        raise SystemExit  

"""Construct RNN"""
if conf.IS_RNN:
    rnn_net = MyRNN(conf.n_classes,conf.rnnsize,conf.batchsize,conf.maxseq,CNNbase,conf.rnn_input_dim,conf.nlayers)
    models.append(rnn_net)

# Define loss 
if conf.IS_RNN:
    cost,preds = models[-1].calc_cost()
else:
    cost,batch_acc,preds = models[-1].calc_cost()

"""Define Optimization settings"""

max_grad_norm = 4
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
        
##get variables e
temp = set(tf.all_variables())

optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

print("[train_script]: Optimization is set")

if conf.IS_RNN:
    if not conf.is_cnn_trainable:
##remaining variables = ADAM variables + trainable variables
        remaining_vars = (set(tf.all_variables()) - temp ) 
        init2 = tf.initialize_variables(remaining_vars)
#init2 = set(tf.all_variables())
    else:
        init2 = tf.initialize_all_variables()

if not conf.IS_RNN:
    remaining_vars = (set(tf.all_variables()) - temp ) 
    init2 = tf.initialize_variables(remaining_vars)

saver = tf.train.Saver(tf.all_variables())

sess.run(init2)

print("[train_script]: Initialized")

step = 1
 
if (conf.IS_RNN==1) and (conf.LOAD_RNN==1):
    ckpt = tf.train.get_checkpoint_state(conf.checkpoint_dir)
    print(ckpt.model_checkpoint_path)    
    print(conf.checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        print("[train_script]: LOADED RNN")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("[train_script]: failed to load")
        raise SystemExit
    last_checkpoint_path = ckpt.model_checkpoint_path
    step = 1+int(last_checkpoint_path[last_checkpoint_path.rindex('-')+1:])
    print(step)

   
# Keep training until reach max iterations
tmp_train_acc = []
tmp_test_acc = []
train_cost = []

reader._ignoresmall = conf.ignoresmall

print(conf.isTest)  

if not conf.isTest:
    print("[train_script]: Start training")
    while step < conf.training_steps:
       
        if conf.IS_RNN:       
            seqlen,seq_xs, seq_ys = reader.read_batch(conf.batchsize)
        else:
            seq_xs, seq_ys = reader.read_batch(conf.batchsize)
    
        if conf.IS_RNN:
            w = rnn_net.weighting(seqlen,[],'uniform')
                    
            """seq_xs: maxseq(outer) * batchsize(inner) X img_dim """
            """seq_ys: batchsize X maxseq """
            """w     : batchsize X maxseq """
            """seqlen: batchsize X 1      """
            
            # Fit training using sequence data
            sess.run( [train_op], feed_dict={rnn_net.x: seq_xs, rnn_net.y: seq_ys, 
                                           rnn_net.keep_prob: conf.dropout, rnn_net.early_stop: seqlen, 
                                           rnn_net.cost_w: w})  
        else:
            sess.run( [train_op], feed_dict={models[-1].x: seq_xs, models[-1].y: seq_ys, 
                                models[-1].keep_prob: conf.dropout})  
             
             
        """"""
        """get training and test error"""
        if step % 2 == 0:
            
            if conf.IS_RNN:
                pr,corr_predictions,cst = models[-1].predict(sess,seq_xs,1.,seqlen,w,seq_ys)   
                tmp_train_acc.extend(corr_predictions)
                train_cost.append(cst)
            else:
                bacc = sess.run( [batch_acc], feed_dict={models[-1].x: seq_xs, models[-1].y: seq_ys, 
                            models[-1].keep_prob: 1.}) 
                tmp_train_acc.extend(bacc)
    
        if step % conf.display_step == 0:
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
        if step % (10*conf.display_step) == 0 and conf.SAVE_MODEL:
            print("[train_script]: checkpoint")
            checkpoint_path = os.path.join(conf.checkpoint_dir, conf.checkpoint_file)
            saver.save(sess, checkpoint_path, global_step=step)        
        step += 1
        
        
    train_err_file.close()
    
    reader.terminate = True
    print("[train_script]: Optimization Finished!")
    if conf.SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(conf.checkpoint_dir, conf.checkpoint_file)
        saver.save(sess, checkpoint_path, global_step=step)


else:
    print("[train_script]: Testing")
    
    step=0
    summ = 0
    reader.index = 0
    
    tmp_train_acc = []
    predictions_file = open(conf.logits_file, "w+")

    if conf.IS_CNN:
        conf.batchsize = 1
        
    while reader._epochs < 1:
    
        seqlen_test,seq_xs_test, seq_ys_test = reader.read_batch(conf.batchsize)
       
        if conf.IS_RNN:    
            w = rnn_net.weighting(seqlen_test)
            pr,corr_predictions,_ = rnn_net.predict(sess,seq_xs_test,1.,seqlen_test,w,seq_ys_test)
        else: 
            logits,seq_prediction, corr_predictions, cost  = models[-1].sequence_predict(sess,seq_xs_test,1.,
                                                                        seqlen_test,seq_ys_test)
        predictions_file.write(""+str(logits[0])+","+str(logits[1])+"\n")

        tmp_train_acc.extend(corr_predictions)
        if step % (2*conf.display_step) == 0:
            print("partial mean acc: " + "{:.3f}".format(np.mean(np.array(tmp_train_acc))))
            tmp_train_acc = []
        summ += np.mean(corr_predictions)
        step += 1
    
    predictions_file.close()
    
    print("Testing Accuracy:", "{:.5f}".format(summ/step))

