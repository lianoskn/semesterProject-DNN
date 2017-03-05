
import os
from scipy import misc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import csv
import numpy as np

from sequenceReader import SequenceReader

LOAD_CNN_checkpoint_path = "/home/lianos91/Desktop/training_patches/model9_2_125/"

USE_CNN = True

dirpath = "/home/lianos91/Desktop/training_patches/train_patches_128_64/"
datapath = "val/"
testreader = SequenceReader(dirpath,datapath,istest=True,maxsequence=16,isComposite=False)

"""Construct and load pretrained CNN model"""
if USE_CNN:
    from myCNN import MyCNN
    import tensorflow as tf
    CNNmodel = MyCNN(2,istrainable=True)
    cost,batch_acc,preds = CNNmodel.calc_cost()
    
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)
    
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

f1 = open("predictions_CNN.txt", "r", newline='\n')
f2 = open("predictions_RNN.txt", "r", newline='\n')

readerModel1 = reader = csv.reader(f1)
readerModel2 = reader = csv.reader(f2)

predsM1 = []
for row in readerModel1:
    #print(row[0])
    predsM1.append(row[0]=='True')
    
predsM2 = []
for row in readerModel2:
    predsM2.append(row[0]=='True')
    

ind = 0    
viz = 0
testreader._ignoresmall = 4
if not USE_CNN:
    testreader.mean_img = np.zeros([64,64])

plt.axis('off')    

numsame = 0


while testreader._epochs < 1:
#while ind < len(predsM1):
    seqlen_test, seq_xs_test, seq_ys_test = testreader.read_batch(1)  
    label = seq_ys_test[0][0]
    
    #seqlen_test, seq_xs_test, seq_ys_test = 1,1,1#testreader.read_batch(1)  
    #label = 1#seq_ys_test[0][0]
    
    if (predsM2[ind] != label) and (predsM1[ind] == label):
        viz = 1
    #    numsame = numsame + 1
    #if label == 1:
    #    viz = 1
    #if predsM2[ind] == predsM1[ind]:
    #    numsame = numsame + 1
        
    if USE_CNN:
        seq_ys = seq_ys_test[0,0:seqlen_test[0]]
        seq_xs = seq_xs_test[0:seqlen_test[0],:]
        conv_codes, d1_codes = sess.run( [CNNmodel.conv5, CNNmodel.dense1], 
                                         feed_dict={CNNmodel.x: seq_xs, 
                                         CNNmodel.keep_prob: 1.})
        
        #w1 = sess.run( [CNNmodel.weights['wc1']], 
        #                                 feed_dict={CNNmodel.x: seq_xs, 
        #                                 CNNmodel.keep_prob: 1.})
        #break        

    if viz:
        sx = seq_xs_test
        
        if USE_CNN:
            dx = d1_codes
        
        ff=plt.figure(1,figsize=(2,16))
        gs1 = gridspec.GridSpec(2,16)
        gs1.update(wspace=0.1, hspace=0) # set the spacing between axes. 

        for i in range(0,seqlen_test[0]):         
           
            a = ff.add_subplot(gs1[0,i])
            #aa.axis('off')
            a.set_xticklabels([])
            a.set_yticklabels([])
            #a.set_aspect('equal')
            plt.imshow(np.reshape(sx[i,:],[64,64]),cmap='Greys_r')
            
            if USE_CNN:
                a = ff.add_subplot(gs1[1,i])
                #aa.axis('off')
                a.set_xticklabels([])
                a.set_yticklabels([])
                #a.set_aspect('equal')
                plt.imshow(np.reshape(dx[i,:],[20,20]),cmap='Greys_r')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        #ff.savefig("compare_rnncnn_img"+str(ind)+".png",bbox_inches='tight')
        #break
    
    ind = ind + 1
    viz = 0    
    
print(numsame)
print(ind)    
    
    
f1.close()
f2.close()