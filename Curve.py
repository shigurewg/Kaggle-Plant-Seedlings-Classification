import numpy as np 
from matplotlib import pyplot as plt 

fd=open('train.txt','r')
fd2=open('val.txt','r')
train_loss=[]
train_acc=[]
val_loss=[]
val_acc=[]
ST=fd.readline()
ST2=fd2.readline()
while ST:
    a,b=ST.split()
    c,d=ST2.split()
    train_loss.append(a)
    train_acc.append(b)
    val_loss.append(c)
    val_acc.append(d)
    ST=fd.readline()
    ST2=fd2.readline()
train_loss=np.array(train_loss,dtype=float)
train_acc=np.array(train_acc,dtype=float)
val_loss=np.array(val_loss,dtype=float)
val_acc=np.array(val_acc,dtype=float)

x=np.arange(1,1+len(train_loss))
fig,ax = plt.subplots(figsize=(16,9))
plt.title('Loss Curve')
plt.xlabel('epocp')
plt.ylabel('Loss')
ax.plot(x,train_loss,label='train') 
ax.plot(x,val_loss,label='valid') 
plt.legend(fontsize=20)
plt.savefig('loss_curve.png')
#acc
y=x
fig,ay = plt.subplots(figsize=(16,9))
plt.title('Accuracy curve')
plt.xlabel('epocp')
plt.ylabel('Accuracy')
ay.plot(x,train_acc,label='train') 
ay.plot(x,val_acc,label='valid') 
plt.legend(fontsize=20)
plt.savefig('accuracy_curve.png')