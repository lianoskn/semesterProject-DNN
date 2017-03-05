import csv
import numpy as np

import matplotlib.pyplot as plt

from sequenceReader import SequenceReader

dirpath = "/home/lianos91/Desktop/training_patches/train_patches_128_64_rich/"
datapath = "val/"
testreader = SequenceReader(dirpath,datapath,istest=True,maxsequence=16,isComposite=True)

print("dataset ready.")

USE_LOGITS = 1
if USE_LOGITS:
    #f1 = open("logits/logits_compositeflowRNN.txt", "r", newline='\n')
    #f2 = open("logits/logits_compositeImgRNN.txt", "r", newline='\n')
    f1 = open("logits/logits_composite_imgflow_fusedCNN2layersd1_end2end.txt", "r", newline='\n')
    f2 = open("logits/logits_composite_fusedCNNfc62layers.txt", "r", newline='\n')
    #f1 = open("logits/logits_CNN.txt", "r", newline='\n')
    #f2 = open("logits/logits_RNN.txt", "r", newline='\n')

else:
    f1 = open("predictions_CNN.txt", "r", newline='\n')
    f2 = open("predictions_RNN.txt", "r", newline='\n')

readerModel1 = csv.reader(f1)
readerModel2 = csv.reader(f2)

predsM1 = []
predsM2 = []

if USE_LOGITS:
    """will contain the probability of label 1"""
    for row in readerModel1:
        #print(row[0])
        f = float(row[1])
        p = np.exp(f)/(1+np.exp(f))
        f = float(row[0])
        p2 = np.exp(f)/(1+np.exp(f))
        predsM1.append(p/(p+p2))
                
    for row in readerModel2:
        f = float(row[1])
        p = np.exp(f)/(1+np.exp(f))
        f = float(row[0])
        p2 = np.exp(f)/(1+np.exp(f))
        predsM2.append(p/(p+p2))
else:
    for row in readerModel1:
    #print(row[0])
        predsM1.append(row[0]=='True')
    
    for row in readerModel2:
        predsM2.append(row[0]=='True')

#predsM2[len(predsM1)::] = []

testreader._ignoresmall = 4

numsame = 0
testreader._index = 0
testreader._epochs = 0

llabels = []
truelen = []
while testreader._epochs < 1:
    seqlen_test, seq_xs_test, seq_ys_test = testreader.read_batch(1)  
    lbl = seq_ys_test[0][0]
    llabels.append(lbl)
    truelen.append(testreader.trueseqlen)

ind = 0    
viz = 0
confusion_matrixM2 = np.zeros([2,2])
confusion_matrixM1 = np.zeros([2,2])
confusion_matrixCNNfRNN = np.zeros([2,2]) #fused 
hsize = 61
histerrs = np.zeros([hsize])

"""1: RNN, 2: CNN"""
while ind < len(llabels):
    label = llabels[ind]
    #if USE_LOGITS:
        #pred_cnnfrnn = predsM2[ind]-0.5 + predsM1[ind]-0.5 > 0
        #pred_cnnfrnn = 0.5*predsM2[ind] + 0.5*predsM1[ind] > 0.5
        #confusion_matrixCNNfRNN[label,pred_cnnfrnn] = confusion_matrixCNNfRNN[label,pred_cnnfrnn] + 1
        
    confusion_matrixM2[label,predsM2[ind]>0.5] = confusion_matrixM2[label,predsM2[ind]>0.5] + 1
    confusion_matrixM1[label,predsM1[ind]>0.5] = confusion_matrixM1[label,predsM1[ind]>0.5] + 1
    
    #if ((predsM2[ind]>0.5) == label) and ((predsM1[ind]>0.5) != label):
    #    viz = 1
        #print(testreader.trueseqlen)
    #    truelen[ind] = np.min([truelen[ind], hsize-1])
    #    histerrs[truelen[ind]] = histerrs[truelen[ind]] + 1   
    
    ind = ind + 1

    
y = plt.figure(1)
rects1 = plt.bar(np.arange(61), histerrs, .42, color='r')
plt.show()  
print("img conf matrix:")    
print(confusion_matrixM2)
print( (confusion_matrixM2[0,0]+confusion_matrixM2[1,1])/np.sum(confusion_matrixM2) )    

print("flow conf matrix:")
print(confusion_matrixM1)  
print( (confusion_matrixM1[0,0]+confusion_matrixM1[1,1])/np.sum(confusion_matrixM1) )    
  
print("Late-fused conf matrix:")
print(confusion_matrixCNNfRNN)           
print( (confusion_matrixCNNfRNN[0,0]+confusion_matrixCNNfRNN[1,1])/np.sum(confusion_matrixCNNfRNN) )    

f1.close()
f2.close()
