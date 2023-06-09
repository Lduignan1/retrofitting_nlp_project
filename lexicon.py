import string
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
  elif any(char not in string.ascii_letters for char in word):
    return '*SYMBOL*'
  # all other words are returned intact
  else:
    return word 
  

class Lexicon:
    def __init__(self, lang='eng'):
        self.words = set()
        self.wn_syn = {}
        self.wn_all = {}
        self.ppdb = defaultdict(set)    # is defaultdict better than normal dict? Why?
        self.lang = lang

        

    ''' WORDNET '''

    ''' populate self.words with all the unique words in the WordNet library (internal method)'''
    def _get_all_words(self):
        if self.lang == 'eng':
            self.words.update(wn.all_lemma_names())
        else:
            self.words.update(wn.all_lemma_names(lang=self.lang))
        return self.words


    ''' get synonymy relations (public method)'''
    def wn_synonyms(self):
        for word in self._get_all_words():
            # normalize word
            word = normalize(word)

            # add a new key for the current word in the self.wn_syn dictionary
            self.wn_syn[word] = []

            synsets = wn.synsets(word, lang=self.lang)
            for syn in synsets:
                for lemma in syn.lemmas():
                    if lemma.name() != word and lemma.name() not in self.wn_syn[word]: 
                        self.wn_syn[word].append(normalize(lemma.name()))
        
        return self.wn_syn


    ''' get the synonymy, hypernymy and hyponymy relations of word (public method)'''
    def wn_all_relations(self):
        # create a copy of the synonym relations dictionary
        synonyms = self.wn_synonyms()
        self.wn_all = synonyms.copy()
        
        # add hypernymy and hyponymy relations
        for word in self.wn_all:
            synsets = wn.synsets(word, lang=self.lang)
            for synset in synsets:
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

    ''' PPDB (public method)'''
    
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
    


### Checks ###

# lexicon = Lexicon()

# # PPDB 
# print(f"PPDB relations: \n{lexicon.read_ppdb('lexicons/ppdb-2.0-xl-lexical')}")

# # WordNet synonyms
# print(f"\nWordNet synonyms: \n{lexicon.wn_synonyms()}")

# # WordNet all_relations
# print(f"\nWordNet synonyms, hypernyms & hyponyms: \n{lexicon.wn_all_relations()}\n")

