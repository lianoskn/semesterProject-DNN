import os
import numpy as np
import tensorflow as tf
from sequenceReader import SequenceReader
from encoderDecoderRNN import EncoderDecoderRNN
from tensorflow.models.rnn import seq2seq
from myRNN import weighting

LOAD_MODEL = True
SAVE_MODEL = True
checkfile = "enc_2048_im32_lrd00008.ckpt"
encoder_checkpoint_path = "/home/lianos91/Desktop/training_patches/Encoder_2048_i32_lrd00008/"
encoderClassifer_checkpoint_path = "/home/lianos91/Desktop/training_patches/encoderRNN_2048_lrd00008/"

""" Learning Parameters"""
learning_rate = .00008
training_steps = 700000
display_step = 100

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

dirpath = "/home/lianos91/Desktop/training_patches/train_patches_128_64/"
datapath = "train_16/"
reader = SequenceReader(dirpath,datapath,istest=True,
                             maxsequence=encoder_length,imdim=img_dim,isComposite=False)

reader._ignoresmall = 5


encoder = EncoderDecoderRNN(n_classes,rnnsize,batchsize,maxseq,img_dim*img_dim,nlayers,encoder_length)
#  __init__(self, nclasses,rnnsize,batchsize,maxseq,inputdim,nlayers = 1):

sess = tf.InteractiveSession()
saver_enc = tf.train.Saver(tf.all_variables())

init1 = tf.initialize_all_variables()
sess.run(init1)

"""Load encoder model"""
ckpt = tf.train.get_checkpoint_state(encoder_checkpoint_path)
print(ckpt.model_checkpoint_path)

if ckpt and ckpt.model_checkpoint_path:
    print("[train_script]: LOADED RNN")
    saver_enc.restore(sess, ckpt.model_checkpoint_path)
else:
    print("[train_script]: failed to load")
    raise SystemExit
last_checkpoint_path = ckpt.model_checkpoint_path
    
cost_enc = encoder.calc_cost()


cost_w = tf.placeholder(tf.float32, [1,encoder_length]) 

ylabel = tf.placeholder(tf.int64, [batchsize, encoder_length])
"""Complement with a softmax. Batch size = 1"""
output = encoder.encoder_outputs

output = tf.reshape(tf.concat(1, output), [-1, rnnsize])
"""outputs: batchsize(outer index) * maxseq(inner index) X inputdim"""

softmax_w = tf.get_variable("softmax_w", [rnnsize, 2])
softmax_b = tf.get_variable("softmax_b", [2])
logits = tf.matmul(output, softmax_w) + softmax_b

yy = tf.reshape(ylabel, [-1])
loss = seq2seq.sequence_loss_by_example(logits=[logits],
                                        targets=[yy],
                                        weights=[tf.reshape(cost_w, [-1])],
                                        average_across_timesteps=False)

costSoftmax = tf.reduce_sum(loss) / 1   
#self.cost =  tf.reduce_sum(loss) / tf.reduce_sum(self.cost_w)
predictions = tf.argmax(logits,1)

"""Define Optimization settings"""
max_grad_norm = 5
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(costSoftmax, tvars,aggregation_method=2),
                                      max_grad_norm)
        
##get variables e
temp = set(tf.all_variables())

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, 
                                   use_locking=False, name='Adam')
train_op = optimizer.apply_gradients(zip(grads, tvars))

remaining_vars = (set(tf.all_variables()) - temp ) | set(tf.trainable_variables())
init2 = tf.initialize_variables(remaining_vars)

saverRNN = tf.train.Saver(tf.all_variables())

sess.run(init2)

step = 0

print(step)

z = np.zeros([1,img_dim*img_dim])

tmpcost = []
train_err_file = open("encoder_b1_2048_img32_lrd00008_seq16.txt", "a+")
labels = []

## dummy arrays. required to run the graph, but not used
dec_rec_input = np.zeros([encoder_length,img_dim*img_dim])#np.vstack((z,(enc_input[::-1,:])[0:-1]))
dec_pred_input = np.zeros([maxseq-encoder_length,img_dim*img_dim])#np.vstack((z,seq_xs[encoder_length:-1,:]))
rec_label = np.zeros([encoder_length,img_dim*img_dim])#enc_input[::-1,:]
pred_label = np.zeros([maxseq-encoder_length,img_dim*img_dim])#seq_xs[encoder_length::,:]

w = weighting(1,encoder_length, [encoder_length],weights="uniform")
while step < training_steps:
    seqlen,seq_xs, seq_ys = reader.read_batch(batchsize)
    seq_xs = seq_xs/255.    
    enc_input = seq_xs[0:encoder_length,:]
    
    # Fit training using sequence data
    preds,costt,_,_,t = sess.run( [predictions, costSoftmax, encoder.output_rec, encoder.output_pred, train_op], 
                                feed_dict={encoder.encoder_inputs: enc_input,
                                    encoder.rec_inputs:  dec_rec_input,
                                    encoder.pred_inputs: dec_pred_input,
                                    encoder.labels_rec:  rec_label,
                                    encoder.labels_pred: np.float32(pred_label),
                                    encoder.keep_prob:   dropout,
                                    ylabel: seq_ys,
                                    cost_w: w} 
                                     )  
    
    labels.append(preds==seq_ys[0][0])
    tmpcost.append(costt)
    if costt > 500:
        print(costt)
    if step % (display_step)==0:
        m = np.mean(np.array(tmpcost))
        print("[train_script]: Step: "+str(step)+ ", train_cost {:.3f}".format(m))
        print("mean_train_acc: "+str(np.mean(np.array(labels))))
        labels = []
        tmpcost = []
        train_err_file.write(""+str(m)+","+str(0)+"\n")
    if step % (5*display_step) == 0 and SAVE_MODEL:
        print("[train_script]: checkpoint")
        checkpoint_path = os.path.join(encoderClassifer_checkpoint_path, checkfile)
        saverRNN.save(sess, checkpoint_path, global_step=step)        
    step += 1

if SAVE_MODEL:
    print("[train_script]: checkpoint")
    checkpoint_path = os.path.join(encoderClassifer_checkpoint_path, checkfile)
    saverRNN.save(sess, checkpoint_path, global_step=step) 
    
