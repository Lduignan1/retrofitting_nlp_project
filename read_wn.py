from nltk.corpus import wordnet as wn

class Wordnet:
    def __init__(self, lang='eng'):
        self.syn = {}
        self.all = {}
        self.lang = lang


    ''' retrieve synsets for specified language'''
    def get_synsets(self, word):
        if self.lang == 'eng':
            return wn.synsets(word)
        else:
            return wn.synsets(word, lang=self.lang)


    ''' get the synonymy relations of word'''
    def synonyms(self, word):
        # add a new key for the current word in the self.syn dictionary
        self.syn[word] = []

        synsets = self.get_synsets(word)
        for syn_word in synsets:
            for lemma in syn_word.lemmas():
                if lemma.name() != word and lemma.name() not in self.syn[word]: 
                    self.syn[word].append(lemma.name())
        
        return self.syn[word]


    ''' get the synonymy, hypernymy and hyponymy relations of word'''
    def all_relations(self, word):
        # initialize dict with the list of synonyms
        self.all[word] = self.synonyms(word)
        
        # add hypernymy and hyponymy relations
        synsets = self.get_synsets(word)
        for synset in synsets:
            # add hypernyms
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    if lemma.name() != word and lemma.name() not in self.all[word]:
                        self.all[word].append(lemma.name())
            # add hyponyms
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    if lemma.name() != word and lemma.name() not in self.all[word]:
                        self.all[word].append(lemma.name())
        
        return self.all[word]
