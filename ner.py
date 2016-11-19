from os import listdir, stat
from os.path import isfile, join

import nltk
from nltk.corpus import PlaintextCorpusReader

mypath="/home/brendan/work/nlp/ner/wsj_untagged"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and stat(mypath + "/" + f).st_size != 0]

corpus = PlaintextCorpusReader(mypath,onlyfiles) 

posTagged = nltk.pos_tag(corpus.words('wsj_1001.txt'))

print(posTagged)
