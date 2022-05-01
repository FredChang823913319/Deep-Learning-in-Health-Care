import codecs
from collections import defaultdict
import csv
import string
from stop_words import get_stop_words    # download stop words package from https://pypi.org/project/stop-words/
import numpy as np

# Code authors: https://github.com/tiantiantu/KSI
# Comments & Refactoring contributor: Haocheng Zhang (hz46@illinois.edu)

# /////////////////INITIALIZATION//////////////////
# a list of english stop words to be filtered
stop_words = get_stop_words('english')
# a dictionary of lists for storing discharge pairs (key: admission id, val: [summary text])
admidic=defaultdict(list)
# count var to track total num of discharge summaries
count=0
# /////////////////END INITIALIZATION//////////////////


# NOTEEVENTS.csv contains following columns:
# ROW_ID, 
# SUBJECT_ID, 
# HADM_ID, 
# CHARTDATE, 
# CHARTTIME, 
# STORETIME, 
# CATEGORY, 
# DESCRIPTION, 
# CGID, 
# ISERROR, 
# TEXT

# build pairs of discharge summary with admission id as keys while tracking total nums of summaries
# for example: 
# map<01, "this is a summary example">, count = 1
# map<02, "this is a summary example2">, count = 2
with open('NOTEEVENTS.csv', 'r') as csvfile:
     csvReader = csv.reader(csvfile, delimiter=',', quotechar='"')
     for row in csvReader:
         if row[6]=='Discharge summary':
             admidic[row[2]].append(row[-1].replace('\n',' ').translate(str.maketrans('','',string.punctuation)).lower())
             count=count+1


# build bags-of-words feature maps
# Note: here, we could potentially have digits
word_vec=defaultdict(int)
for adm_id in admidic:
    for text in admidic[adm_id]:
        words=text.strip('\n').split()
        for adm_id in words:
            word_vec[adm_id]=word_vec[adm_id]+1


# filter the word_vec with following conditions:
# the word is NOT a digit
# the word has occurances > 10
# the word is NOT a stop word
filtered_word_vec=defaultdict(int)
for adm_id in word_vec:
    if adm_id.isdigit()==False:
        if word_vec[adm_id]>10:
            if adm_id not in stop_words:
                filtered_word_vec[adm_id]=word_vec[adm_id]

# DIAGNOSES_ICD.csv contains following columns:
# ROW_ID, 
# SUBJECT_ID, 
# HADM_ID, 
# SEQ_NUM, 
# ICD9_CODE

file1=codecs.open('DIAGNOSES_ICD.csv','r')
ad2c=defaultdict(list)
text=file1.readline() # skip title
text=file1.readline()

# build ICD_9 feature maps with admission id as keys
while text:
    text=text.strip().split(',')
    # an example of data: # ['1297', '109', '172335', '1', '"40301"']
    # notice a pair of quotation marks in ICD_9 codes.
    # below snippet extract a ICD_9 code and build feature maps.
    if text[4][1:-1]!='': 
        ad2c[text[2]].append("d_"+text[4][1:-1])
    text=file1.readline()

code_vec=defaultdict(int)
for adm_id in ad2c:
    for code in ad2c[adm_id]:
        code_vec[code]=code_vec[code]+1


# this is code threshold for code frequency, here we use 0 to include all codes
code_threshold=0
# this is the target file we want to output
file_out=codecs.open("combined_dataset",'w')

# IDlist.py contains list of admission ids, like ['001', '002', ...]
IDlist=np.load('IDlist.npy',encoding='bytes').astype(str)

# below output admission id with filtered codes
for adm_id in IDlist:
    if ad2c[adm_id]!=[]:
        
        file_out.write('start! '+adm_id+'\n')
        file_out.write('codes: ')
        temp_codes=[]
        for code in ad2c[adm_id]:
            if code_vec[code]>=code_threshold:
                if code[0:5] not in temp_codes:
                    temp_codes.append(code[0:5])
       
        for code in temp_codes:
            file_out.write(code+" ")
        file_out.write('\n')
        file_out.write('notes:\n')
        for text in admidic[adm_id]:    
            words=text.strip('\n').split() 
            for word in words:
                if filtered_word_vec[word]!=0:
                    file_out.write(word+" ")
            file_out.write('\n')
        file_out.write('end!\n')
file_out.close()

