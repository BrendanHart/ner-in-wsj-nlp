from os import listdir, stat
from os.path import isfile, join

from enamexparser import *
from enamex_bayes_classifier import *
from test import *

import nltk
from nltk.corpus import PlaintextCorpusReader

import wikification

mypath="/home/brendan/work/nlp/ner/wsj_testdata"

class NERTagger():

    def __init__(self):
        self.classifier = EnamexBayesClassifier()

    def tag(path, files):
        corpus = PlaintextCorpusReader(path, files)
        count = 1
        taggedDocs = {}
        nes = {}
        for f in files:
            print("Tagging file " + f + ", " + str(count) + "/" + str(len(files)))
            count += 1

            chunked = process(corpus.raw(f))
            for i in range(len(chunked)):
                for j in range(len(chunked[i])):
                    chunked[i][j] = neTagSentence(chunked[i][j])
            taggedDocs[f] = chunked

            foundNes = []
            for split in chunked:
                for sentence in split:
                    foundNes += getNesFromTree(sentence)
            nes[f] = foundNes
        return (taggedDocs, nes)
             
    def train(trainDir, trainFiles):

        self.parser = EnamexParser(trainDir, trainFiles)
        count = 0
        for f in trainFiles:
            count += 1
            parsed = self.parser.get(f)
            print("Training: " + f + " - " + str(count)+"/"+str(len(trainFiles)))
            self.classifier.train(EnamexParser.neTreesFromParsed(parsed))

    def test(trainDir, trainFiles, testDir, testFiles, looseTest=False):
        print("NOTE: The following training has no effect on the tag function. It is merely for testing.")

        oldClassifier = self.classifier
        oldParser = self.parser
        self.classifier = EnamexBayesClassifier()

        self.train(trainDir, trainFiles)
        (taggedDocs, nes) = self.tag(testDir, testFiles)

        percSum = 0
        for f in testFiles:
            perc = 100 * Test.looseCalculate(nes[f], EnamexParser.nesFromParsed(self.parser.get(f)))
            percSum += perc
            print(f, perc)

        self.classifier = oldClassifier
        self.parser = oldParser

        print("Average: " + str(percSum / (len(testFiles))))
        

def read(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and stat(path + "/" + f).st_size != 0]

    return PlaintextCorpusReader(path,onlyfiles) 

def process(raw):
    sents = []
    
    # Split at new lines since otherwise things such as Ltd.\nBob are the same sentence
    for s in raw.split("\n"):
        sents.append(nltk.sent_tokenize(s))
    for i in range(len(sents)):
        for j in range(len(sents[i])):
            sents[i][j] = nltk.pos_tag(nltk.word_tokenize(sents[i][j]))
            for k in range(len(sents[i][j])):
                if sents[i][j][k][0] == "&":
                    sents[i][j][k] = (sents[i][j][k][0], "AMP")
                if sents[i][j][k][0] == ".":
                    sents[i][j][k] = (sents[i][j][k][0], "PERIOD")
            sents[i][j] = chunk(sents[i][j])

    return sents

def stitch(sentences):

    for i in range(len(sentences)):
        sentences[i] = flatten(sentences[i])
    sentences = "\n".join(sentences)
    endOfSentPunct = [")","%",",",".","!","?",":",";"]
    count = 0

    sentences = list(sentences)
    i = 0 
    while i < (len(sentences)-1):
        if(sentences[i] == " "):
            if(sentences[i+1] in endOfSentPunct):
                sentences[i] = sentences[i+1]
                sentences.pop(i+1)
        i+=1

    endOfSentPunct = ["`", "£", "(", "$"] 
    i=1
    while i < (len(sentences)):
        if(sentences[i] == " "):
            if(sentences[i-1] in endOfSentPunct):
                sentences[i] = sentences[i-1]
                sentences.pop(i-1)
        i+=1
    
    sentences = fixQuotes(sentences)

    return ''.join(sentences)

def fixQuotes(text):
    seenSingle = False
    seenDouble = False

    i = 0
    while i  < len(text):
        if text[i] == "\'":
            text[i] = "'"
            if seenSingle:
                if text[i-1] == " " and i-1 >= 0:
                    text.pop(i-1)
                    i-=1
                seenSingle = False
            else:
                if i+1 < len(text):
                    if text[i+1] == " ":
                        text.pop(i+1)
                seenSingle = True
        elif text[i] == '\"':
            text[i] = '"'
            if seenDouble:
                if text[i-1] == " " and i-1 >= 0:
                    text.pop(i-1)
                    i-=1
                seenDouble = False
            else:
                if i+1 < len(text):
                    if text[i+1] == " ":
                        text.pop(i+1)
                seenDouble = True 

        i+=1

    return text


def chunk(posTagged):
    ### NOTE: this grammar picks up chairman of Company
    ###       as well as United Sates
    ### Review CC, Moody's and S&P != 1 entity
    ### Picks up group

    grammar = """
    NE: {<NNP.?>+}
    NE: {(<NNP.?>|<NE>)<PERIOD><NE>}
    NE: {<NE><AMP><NE>}
    NE: {<NE><POS><NE>}
    NE: {<NE><IN><NE>}
    """

    cp = nltk.RegexpParser(grammar)

    parsed = cp.parse(posTagged)

    return parsed

### NOTE Potentially redundant code, not used anywhere
def treeToSent(t):
    sent = ""
    if(type(t) is nltk.Tree):
        if t.label() in ["ORGANISATION", "PERSON", "LOCATION"]:
            sent += " <ENAMEX TYPE=\""+t.label()+"\">"+flatten(t)+"</ENAMEX>"
        else:
            for c in t:
                sent += treeToSent(c)
    else:
        sent += " " + t[0]
    return sent
###

def neToTag(label, t):
    res = "<ENAMEX TYPE=\"" + label + "\">"
    for c in t: 
        res += c + " "
    return (res[:len(res)-1] + "</ENAMEX>")

def extractWordTag(t):
    r = []
    for i in t:
        if type(i) is nltk.Tree:
            for w in extractWordTag(i):
                r.append(w)
        else:
            r.append(i)
    return r

def flatten(t):
    phrase = ""
    for c in t:
        if(type(c) is nltk.Tree):
            classes = ["ORGANIZATION","LOCATION","PERSON"]
            if(c.label() in classes):
                phrase += " " + neToTag(c.label(), c)    
            else:
                phrase += " " + flatten(c)
        else:
            print(c)
            if c[1] == "POS":
                phrase += c[0]
            else:
                phrase += " " + c[0]
    return phrase[1:]

corpus = read(mypath)

def getFirstWord(t):
    if type(t) is nltk.Tree:
        return getFirstWord(t[0])
    else:
        return t

### TODO move bayes classiying into classify
def neTagSentence(s):
    for i in range(len(s)):
        if type(s[i]) is nltk.Tree:
            if(s[i].label() == "NE"):
                flattened = flatten(s[i])
                result = classify(flattened)
                if(result == None):
                    if i+1 >= len(s):
                        result = classifier.classifyNE(extractWordTag(s[i]), None)
                    else:
                        w = getFirstWord(s[i+1])
                        result = classifier.classifyNE(extractWordTag(s[i]), w)
                s[i] = nltk.Tree(result, [flattened])

    return s
    
                
def classify(ne):
    locRE = ["\d+\s[A-Za-z]+", "[A-Za-z]+\s\d+"]
    for r in locRE:
        if re.match(r, ne.lower()):
            return "LOCATION"

    orgRE = [".*inc.*", ".*corp.*", ".*ltd.*"]
    for r in orgRE:
        if re.match(r, ne.lower()):
            return "ORGANIZATION"

    return wikification.wikiLookup(ne)

    #for n in NAMES:
    #    if n in ne.split(" "):
    #        return "PERSON"


def getNesFromTree(tree):
    nes = []
    for e in tree:
        if type(e) is nltk.Tree:
            if(e.label() != "S"):
                ne = ""
                for w in e:
                    ne += " " + w
                nes.append((e.label(), ne[1:]))
    return nes


path = "./wsj_test_tagged/"

trainFiles = [f for f in listdir(path) if isfile(join(path, f))][:1] #and stat(path + f).st_size != 0][:700]
testFiles = [f for f in listdir(path) if isfile(join(path, f)) and not f in trainFiles][:1]
allFiles = trainFiles + testFiles

parser = EnamexParser(path, allFiles)
count = 0
for f in trainFiles:
    count += 1
    parsed = parser.get(f)
    print("training " + str(count))
    classifier.train(EnamexParser.neTreesFromParsed(parsed))

taggedDocs = {}
nes = {}
count = 0
for f in testFiles:
    print("test file: " + str(count))
    count+=1
    chunked = process(corpus.raw(f))
    for i in range(len(chunked)):
        for j in range(len(chunked[i])):
            chunked[i][j] = neTagSentence(chunked[i][j])
    taggedDocs[f] = chunked

    foundNes = []
    for split in chunked:
        for sentence in split:
            foundNes += getNesFromTree(sentence)
    nes[f] = foundNes

percSum = 0
for f in testFiles:
    perc = 100 * Test.looseCalculate(nes[f], EnamexParser.nesFromParsed(parser.get(f)))
    percSum += perc
    print(f, perc)
print("Average: " + str(percSum / (len(testFiles))))
