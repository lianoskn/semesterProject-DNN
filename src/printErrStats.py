
import csv
import matplotlib.pyplot as plt
import numpy as np


reader = csv.reader(open("encoder_b1_2048_img32_lrd00008_seq16.txt", newline='\n'), delimiter=',')

#reader = csv.reader(open("train_err_d1_2x64_end2end.txt", newline='\n'), delimiter=',')

"""Structure is:
    (batch1) cost, train accuracy, test acc
    (batch2) ... , ..."""
    
train_acc = []
train_err = []    
testacc = []    
for row in reader:
    train_err.append(float(row[0]))
    train_acc.append(float(row[1]))
    #testacc.append(float(row[2]))
    
y = plt.figure(1)   
plt.subplot(211) 
plt.plot(np.array(train_err),'b')
plt.title('training error') 

plt.subplot(212)
plt.plot(np.array(train_acc),'r')#ls,label="train acc")
#plt.plot(np.array(testacc),'b',label="test acc")
plt.legend(loc='upper left')
#plt.hold()

plt.title('accuracy')  
plt.show()

"""equivalent of 60 display steps with bsize 32."""
"""equivalent of 100 steps with bsize 2."""
#y.savefig('tinyd1.png')

