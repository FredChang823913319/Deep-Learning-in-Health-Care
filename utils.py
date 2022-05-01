import torch
import torch.autograd as autograd
import numpy as np


# same processing method from https://github.com/tiantiantu/KSI, but with params for easy reuse
# rename variables and add comments for better readability... (Haocheng Zhang)
def preprocessing(data, label_to_ix, word_to_ix, wikivoc, batchsize=32):

    new_data=[]
    for doc, note, codes in data:
        templabel=[0.0]*len(label_to_ix)
        for code in codes:
            if code in wikivoc:
                templabel[label_to_ix[code]]=1.0
        templabel=np.array(templabel,dtype=float)
        new_data.append((doc, note, templabel))
    new_data=np.array(new_data)
    
    lenlist=[]
    # data consists of [(doc, note, templabel),...]
    for data in new_data:
        lenlist.append(len(data[0]))
    sortlen=sorted(range(len(lenlist)), key=lambda k: lenlist[k])  
    new_data=new_data[sortlen]
    
    batch_data=[]
    # start batching according to batch_size
    for start_ix in range(0, len(new_data)-batchsize+1, batchsize):
        thisblock=new_data[start_ix:start_ix+batchsize]
        mybsize= len(thisblock)
        numword=np.max([len(data[0]) for data in thisblock])
        main_matrix = np.zeros((mybsize, numword), dtype= np.int)
        for doc in range(main_matrix.shape[0]):
            for codes in range(main_matrix.shape[1]):
                try:
                    if thisblock[doc][0][codes] in word_to_ix:
                        main_matrix[doc,codes] = word_to_ix[thisblock[doc][0][codes]]
                    
                except IndexError:
                    pass       # because initialze with 0, so you pad with 0
    
        notes=[]
        labels=[]
        for data in thisblock:
            notes.append(data[1])
            labels.append(data[2])
        
        notes=np.array(notes)
        labels=np.array(labels)
        batch_data.append((autograd.Variable(torch.from_numpy(main_matrix)),autograd.Variable(torch.FloatTensor(notes)),autograd.Variable(torch.FloatTensor(labels))))
    return batch_data
