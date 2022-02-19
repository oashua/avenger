import numpy as np
import random
import string
def seq2se(seq):
    """
    input: binary sequence
    output: time where edge of input sequence"""
    seq2 = [seq[m+1]-seq[m] for m in range(len(seq)-1)]
    seq2 = np.array(seq2)
    if np.where(seq2==1)[0].shape[0] == 0:
        s = 0
    else:
        s = (np.where(seq2==1)[0][0]+1)/len(seq)
    if np.where(seq2==-1)[0].shape[0] == 0:
        e = 1
    else:
        e = np.where(seq2==-1)[0][0]/len(seq)
    return s,e
def iou(t1,t2):
    if t1[1]<t2[0] or t2[1]<t1[0]:
        return 0
    else:
        if t1[0]<t2[0]:
            l0,l1 = t1[0],t2[0]  
        else:
            l0,l1 = t2[0],t1[0]
        if t1[1]<t2[1]:
            r0,r1 = t1[1],t2[1]  
        else:
            r0,r1 = t2[1],t1[1]
    return (r0-l1)/(r1-l0)

def getKey():
	key=random.sample(string.ascii_letters+string.digits,8)
	keys="".join(key)
	return keys

if __name__ == "__main__":
    seq = np.array([0.1,0.1,0.1,0.6,0.6,0.1,0.6,0.6,0.1,0.1])
    align_seq = list(map(lambda x:1 if x>0.5 else 0,seq))
    s,e = seq2se(align_seq)
    print("{}\n{}".format(s,e))

    print(iou((0.3,0.4),(0.2,0.9)))