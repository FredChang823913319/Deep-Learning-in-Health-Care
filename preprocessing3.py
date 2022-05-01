import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import codecs

# Code authors: https://github.com/tiantiantu/KSI
# Additional Comments & Refactoring contributor: Haocheng Zhang (hz46@illinois.edu)

##########################################################
wikivoc={}
codewiki=defaultdict(list)

file2=codecs.open("wikipedia_knowledge",'r','utf-8')
line=file2.readline()

#relative position for current processing line
relative_pos=0

# 
while line:
    if line[0:4]=='XXXd':
        line=line.strip('\n')
        words=line.split()
        for word in words:
            if word[0:2]=='d_': # check if word is a 'code'
                codewiki[word].append(relative_pos)
                wikivoc[word]=1
        relative_pos=relative_pos+1
    line=file2.readline()

################### four codes have two wikidocuments, correct them
codewiki['d_072']=[214]
codewiki['d_698']=[125]
codewiki['d_305']=[250]
codewiki['d_386']=[219]

np.save('wikivoc',wikivoc)
##################################################
filec=codecs.open("combined_dataset",'r','utf-8')
line=filec.readline()

feature=[]
label=[]
# builds label and features
while line:
    line=line.strip('\n')
    words=line.split()
    
    if words[0]=='codes:':
        temp=words[1:]
        label.append(temp)
        line=filec.readline()
        line=line.strip('\n')
        words=line.split()
        if  words[0]=='notes:':
            tempf=[]
            line=filec.readline()
           
            while line!='end!\n':
                line=line.strip('\n')
                words=line.split()
                tempf=tempf+words
                line=filec.readline()
            feature.append(tempf)
    line=filec.readline()

# initialize prevoc list with position info
# example: {'d_519': 0, 'd_491': 1, 'd_518': 2, 'd_486': 3, 'd_276': 4, 'd_244': 5, 'd_311': 6}
prevoc={}
for codes in label:
    for code in codes:
        if code not in prevoc:
            prevoc[code]=len(prevoc)

##################################
notevec=np.load('notevec.npy')
wikivec=np.load('wikivec.npy')
label_to_ix = {}
ix_to_label={}

# builds label to index map and index to label map
for codes in label:
    for code in codes:
        if code not in label_to_ix:
            label_to_ix[code]=len(label_to_ix)
            ix_to_label[label_to_ix[code]]=code

tempwikivec=[]

for ix in range(0,len(ix_to_label)):
    if ix_to_label[ix] in wikivoc:
        temp=wikivec[codewiki[ix_to_label[ix]][0]]
        tempwikivec.append(temp)
    else:
        tempwikivec.append([0.0]*wikivec.shape[1])
wikivec=np.array(tempwikivec)

####################################

data=[]
for codes in range(0,len(feature)):
    data.append((feature[codes], notevec[codes], label[codes]))
    
data=np.array(data)  

label_to_ix = {}
ix_to_label={}

for doc, note, codes in data:
    for code in codes:
        if code not in label_to_ix:
            if code in wikivoc:
                label_to_ix[code]=len(label_to_ix)
                ix_to_label[label_to_ix[code]]=code

np.save('label_to_ix',label_to_ix)
np.save('ix_to_label',ix_to_label)

training_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
training_data, val_data = train_test_split(training_data, test_size=0.125, random_state=42)

np.save('training_data',training_data)
np.save('test_data',test_data)
np.save('val_data',val_data)


word_to_ix = {}
ix_to_word={}
ix_to_word[0]='OUT'


for doc, note, codes in training_data:
    for word in doc:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)+1
            ix_to_word[word_to_ix[word]]=word  
    
np.save('word_to_ix',word_to_ix)
np.save('ix_to_word',ix_to_word)

newwikivec=[]
for codes in range(0,len(ix_to_label)):
    newwikivec.append(wikivec[prevoc[ix_to_label[codes]]])
newwikivec=np.array(newwikivec)
np.save('newwikivec',newwikivec)

