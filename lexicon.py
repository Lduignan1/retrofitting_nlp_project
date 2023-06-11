import string
import unicodedata as ud    # needs to be mentioned in report
from nltk.corpus import wordnet as wn
from collections import defaultdict


def normalize(word):
    # words containing digits are replaced with '*NUM*'
    if any(char.isdigit() for char in word):
        return '*NUM*'
    # punctuations are replaced with '*PUNC*'
    elif word in string.punctuation:
        return '*PUNC*'
    # words containing symbols other than alphabetical characters, digits or punctuation marks are replaced with '*SYMBOL*'
    # elif any(char not in string.ascii_letters for char in word):
    #     return '*SYMBOL*'
    elif any(ud.category(char) not in ['Ll', 'Lu'] for char in word):
        return '*SYMBOL*'
    # all other words are returned intact
    else:
        return word 
  

class Lexicon:
    def __init__(self, lang='eng'):
        self.wn_syn = {}
        self.wn_all = {}
        self.ppdb = defaultdict(set)    # is defaultdict better than normal dict? Why?
        self.lang = lang
        self.synsets = defaultdict(list)
        

    ''' WORDNET '''

    ''' get synonymy relations '''
    def wn_synonyms(self):
        # get all the unique words in the WordNet library for the specified language
        words = set(wn.all_lemma_names(lang=self.lang))
        
        for word in words:
            # normalize word
            word = normalize(word)

            # add a new key for the current word in the self.wn_syn dictionary
            self.wn_syn[word] = []

            # storing synsets for each word in the self.synsets dictionary to avoid redundent calls to wn.synsets() later on
            self.synsets[word] = wn.synsets(word, lang=self.lang)
            
            for syn in self.synsets[word]:
                for lemma in syn.lemmas():
                    if lemma.name() != word and lemma.name() not in self.wn_syn[word]: 
                        self.wn_syn[word].append(normalize(lemma.name()))
        
        return self.wn_syn


    ''' get the synonymy, hypernymy and hyponymy relations of word '''
    def wn_all_relations(self):
        # create a copy of the synonym relations dictionary
        synonyms = self.wn_synonyms()
        self.wn_all = synonyms.copy()
        
        # add hypernymy and hyponymy relations to it
        for word in self.wn_all:
            for synset in self.synsets[word]:
                # add hypernyms
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        if lemma.name() != word and lemma.name() not in self.wn_all[word]:
                            self.wn_all[word].append(normalize(lemma.name()))
                # add hyponyms
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        if lemma.name() != word and lemma.name() not in self.wn_all[word]:
                            self.wn_all[word].append(normalize(lemma.name()))
        
        return self.wn_all


    ''' PPDB '''
    
    def read_ppdb(self):
        '''
        read a file containing the paraphrase database and return a dictionary
        input: file or path
        output: dict with phrase terms as keys and paraphrases as values
        '''
        if self.lang == 'eng':
            ppdb_file = 'lexicons/ppdb-2.0-xl-lexical'

        else:
            ppdb_file = 'lexicons/ppdb-1.0-xl-lexical'
            
        with open(ppdb_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.split('|||')
            
                # strip() to remove spaces before and after words
                self.ppdb[normalize(line[1].strip())].add(normalize(line[2].strip()))
            
        return self.ppdb
    