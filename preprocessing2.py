import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Code authors: https://github.com/tiantiantu/KSI
# Additional Comments & Refactoring contributor: Haocheng Zhang (hz46@illinois.edu)

# builds wiki vocab feature maps like {'breast': 1, 'cancer': 1, 'is': 1}
# note: it skips processing if line looks like this: 'XXXdiseaseXXX   breast cancer  d_174  d_175\r\n'
wikivocab={}
file1=codecs.open("wikipedia_knowledge",'r','utf-8') 
line=file1.readline()
while line:
    if line[0:3]!='XXX':
        line=line.strip('\n')
        vocabs=line.split()
        for vocabs in vocabs:
            wikivocab[vocabs.lower()]=1
    line=file1.readline()




notesvocab={}
filec=codecs.open("combined_dataset",'r','utf-8')
line=filec.readline()

# builds notes vocab feature maps
while line:
    line=line.strip('\n')
    line=line.split()
    
    if line[0]=='codes:':
        line=filec.readline()
        line=line.strip('\n')
        line=line.split()
        
        if  line[0]=='notes:':
            line=filec.readline()
            while line!='end!\n':
                line=line.strip('\n')
                line=line.split()
                for word in line:
                    notesvocab[word]=1
                line=filec.readline()
    line=filec.readline()


# find common vocabs between notes and wiki
notes_set=set(notesvocab)
wiki_set=set(wikivocab)
notes_wiki_intersection=notes_set.intersection(wiki_set)

# builds wiki documents in list that has common vocabs in notes_wiki_intersection
# example: [['patient'], [], ['patient'], ['patient'], [], ['patient', 'patient', 'patient', 'patient'], [], ['patient'], [], ['patient', 'patient', 'patient', 'patient', 'patient'], [], [], ['patient', 'patient', 'patient', 'patient'], ['patient'], ...]
wikidocuments=[]
file2=codecs.open("wikipedia_knowledge",'r','utf-8')
line=file2.readline()
while line:
    if line[0:4]=='XXXd':
        tempf=[]
        line=file2.readline()
        while line[0:4]!='XXXe':
            line=line.strip('\n')
            line=line.split()
            for vocabs in line:
                if vocabs.lower() in notes_wiki_intersection:
                    tempf.append(vocabs.lower())
            line=file2.readline()
        wikidocuments.append(tempf)
        
    line=file2.readline()

# builds notes documents in list that has common vocabs in notes_wiki_intersection
notesdocuments=[]
file3=codecs.open("combined_dataset",'r','utf-8')
line=file3.readline()

while line:
    line=line.strip('\n')
    line=line.split()
    if line[0]=='codes:':
        line=file3.readline()
        line=line.strip('\n')
        line=line.split()
        
        if  line[0]=='notes:':
            tempf=[]
            line=file3.readline()
        
            while line!='end!\n':
                line=line.strip('\n')
                line=line.split()
                for word in line:
                    if word in notes_wiki_intersection:
                        tempf.append(word)
                
                line=file3.readline()
                
            
            notesdocuments.append(tempf)
    line=file3.readline()

####################################################################################
# initialize notes vocab maps with zeros for later vectorization
notesvocab={}
for vocabs in notesdocuments:
    for vocab in vocabs:
        if vocab.lower() not in notesvocab:
            notesvocab[vocab.lower()]=len(notesvocab)
            
# builds note data that contains a list of document's vocabs.
# ['mg patient patient mg mg mg mg mg mg mg mg mg patient ', '...']
notedata=[]
for vocabs in notesdocuments:
    temp=''
    for vocab in vocabs:
        temp=temp+vocab+" "
    notedata.append(temp)
    
# builds wiki data like note data above
wikidata=[]
for vocabs in wikidocuments:
    temp=''
    for vocab in vocabs:
        temp=temp+vocab+" "
    wikidata.append(temp)    
##########################################################
# vectorize note data in binary format
vect = CountVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
binaryn = vect.fit_transform(notedata)
binaryn=binaryn.A
binaryn=np.array(binaryn,dtype=float)

# vectorize wiki data in binary format
vect2 = CountVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
binaryk = vect2.fit_transform(wikidata)
binaryk=binaryk.A
binaryk=np.array(binaryk,dtype=float)

#save both wiki vec and note vec to files.
np.save('notevec',binaryn)
np.save('wikivec',binaryk)
