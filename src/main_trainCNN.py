
import os
import numpy as np

import tensorflow as tf

#from mySequenceReader import MySequenceReader
from imageReader import ImageReader
from myCNN import MyCNN

LOAD_CNN = True

SAVE_MODEL = True

checkfile = "myCNNadamlr0d0001.ckpt"

CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9_2/"

# Parameters
learning_rate = .0001
training_steps = 25000
display_step = 100

# Network Parameters
n_classes = 2 # classes 
dropout = .5 #0.70 # Dropout, probability to keep units
batchsize = 256

reader = ImageReader("train/",istest=False)

"""Construct CNN model"""

CNNmodel = MyCNN(n_classes,istrainable=True)
cost,batch_accuracy,_ = CNNmodel.calc_cost()

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

saverCNN = tf.train.Saver(tf.all_variables())
init1 = tf.initialize_all_variables()

sess = tf.InteractiveSession()

sess.run(init1)

step = 0

if LOAD_CNN:
    ckpt = tf.train.get_checkpoint_state(CNN_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        print("[train_script]: LOADED CNN!")
        saverCNN.restore(sess, ckpt.model_checkpoint_path)
        last_checkpoint_path = ckpt.model_checkpoint_path
        step = 1+int(last_checkpoint_path[last_checkpoint_path.rindex('-')+1:])
    else:
        print("[train_script]: Failed to LOAD CNN!")
        raise SystemExit  

print("[train_script]: Start training")
   
# Keep training until reach max iterations
tmp_train_acc = []

while step < training_steps:
    
    seq_xs, seq_ys = reader.read_batch(batchsize)
    
    sess.run( [optimizer], feed_dict={CNNmodel.x: seq_xs, CNNmodel.y: seq_ys, 
                            CNNmodel.keep_prob: dropout})  
    
    if step % 5 == 0:
        bacc = sess.run( [batch_accuracy], feed_dict={CNNmodel.x: seq_xs, CNNmodel.y: seq_ys, 
                            CNNmodel.keep_prob: 1.}) 
        tmp_train_acc.extend(bacc)
        if step % (display_step) == 0:
            print("step "+str(step)+". partial acc: " + "{:.3f}".format(np.mean(np.array(tmp_train_acc))))
            tmp_train_acc = []     
            if np.sum(np.isnan(bacc)):
                print("____Diverged____")
                raise SystemExit   
                          
    #checkpoint save
    if step % (5*display_step) == 0 and SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(CNN_checkpoint_path, checkfile)
        saverCNN.save(sess, checkpoint_path, global_step=step)        
        
    step += 1
    
    
reader.terminate = True
print("[train_script]: Optimization Finished!")
if SAVE_MODEL:
    print("[train_script]: checkpoint")
    checkpoint_path = os.path.join(CNN_checkpoint_path, checkfile)
    saverCNN.save(sess, checkpoint_path, global_step=step)
