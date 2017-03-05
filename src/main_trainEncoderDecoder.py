import os
import numpy as np
import tensorflow as tf
from sequenceReader import SequenceReader
from encoderDecoderRNN import EncoderDecoderRNN

LOAD_MODEL = True
SAVE_MODEL = True
checkfile = "enc_2048_im32_lrd00008.ckpt"
save_checkpoint_path = "/home/lianos91/Desktop/training_patches/Encoder_2048_i32_lrd00008/"

""" Learning Parameters"""
learning_rate = .00008
training_steps = 2000000
display_step = 500

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

#reader = MySequenceReader("train_largerThan4/",istest=False,maxsequence=maxseq)
#reader = MySequenceReader("train_tiny/",istest=False,maxsequence=maxseq,imdim=img_dim)

reader = SequenceReader("train_16/",istest=False,maxsequence=maxseq,imdim=img_dim)
reader._ignoresmall = 5

encoder = EncoderDecoderRNN(n_classes,rnnsize,batchsize,maxseq,img_dim*img_dim,nlayers,encoder_length)
#  __init__(self, nclasses,rnnsize,batchsize,maxseq,inputdim,nlayers = 1):

cost = encoder.calc_cost()

"""Define Optimization settings"""
max_grad_norm = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars,aggregation_method=2),
                                      max_grad_norm)
        
##get variables e
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
train_err_file = open("encoder_b1_2048_img32_lrd00008_seq16.txt", "a+")
labels = []
while step < training_steps:
    seqlen,seq_xs, seq_ys = reader.read_batch(batchsize)
    seq_xs = seq_xs/255.    
    enc_input = seq_xs[0:encoder_length,:]
    dec_rec_input = np.vstack((z,(enc_input[::-1,:])[0:-1]))
    dec_pred_input = np.vstack((z,seq_xs[encoder_length:-1,:]))
    
    rec_label = enc_input[::-1,:]
    pred_label = seq_xs[encoder_length::,:]
    # Fit training using sequence data
    costt,_,_,t = sess.run( [cost, encoder.output_rec, encoder.output_pred, train_op], 
                                feed_dict={encoder.encoder_inputs: enc_input,
                                    encoder.rec_inputs:  dec_rec_input,
                                    encoder.pred_inputs: dec_pred_input,
                                    encoder.labels_rec:  rec_label,
                                    encoder.labels_pred: np.float32(pred_label),
                                    encoder.keep_prob:   dropout} 
                                     )  
    labels.append(seq_ys[0][0])
    tmpcost.append(costt)
    if costt > 500:
        print(costt)
    if step % (display_step)==0:
        m = np.mean(np.array(tmpcost))
        print("[train_script]: Step: "+str(step)+ ", train_acc {:.3f}".format(m))
        print("label: "+str(np.mean(np.array(labels))))
        labels = []
        tmpcost = []
        train_err_file.write(""+str(m)+","+str(0)+"\n")
    if step % (5*display_step) == 0 and SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
        saver.save(sess, checkpoint_path, global_step=step)        
    step += 1

if SAVE_MODEL:
    print("[train_script]: checkpoint")
    checkpoint_path = os.path.join(save_checkpoint_path, checkfile)
    saver.save(sess, checkpoint_path, global_step=step) 
    
