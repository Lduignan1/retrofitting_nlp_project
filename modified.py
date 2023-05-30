import gzip
import numpy as np
import sys
import string

from copy import deepcopy


# modified
def norm_word(word):
  # words consisting entirely of digits (also '-' and '.') are replaced with '---num---'
  # (do we also want words that contain digits to be replaced with '---num---'?)
  if word.lstrip('-').replace('.', '').isdigit():
    return '---num---'
  # punctuations are replaced with '---punc---'
  elif word in string.punctuation:
    return '---punc---'
  # all other words are converted to lowercase
  else:
    return word.lower()


# modified
''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
  wordVectors = {}
  with (gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename, 'r')) as file:  # 'rt' = open for reading as text file
    for line in file:
      elements = line.strip().lower().split()
      word = elements[0]
      vector = np.array([float(val) for val in elements[1:]], dtype=float) # is it important to use numpy arrays instead of python lists?
      # normalize vector (Euclidean norm)
      norm = np.linalg.norm(vector)
      # changed from norm + 1e-6 improved results slighlty
      wordVectors[word] = vector / norm  # avoid division by zero if the norm of the vector is zero
    
  print(f"\nVectors read from: {filename}\n")
  return wordVectors


# modified (does the output have to be in plain text format?)
''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  print(f"Writing down the vectors in {outFileName}\n")
  with open(outFileName, 'w') as file:
    # the following could all be done in one line with json.dump(wordVectors, file), but json doesn't support numpy arrays 
    for word, vector in wordVectors.items():
        file.write(f"{word} ")
        np.savetxt(file, vector[np.newaxis], delimiter=' ', fmt='%.4f')    

  
# modified
''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename):
  lexicon = {}  # keys: 1st word of each line - values: [all the other words in the line]
  with open(filename, 'r') as file:
    for line in file:
      words = line.lower().strip().split()
      lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]] 
  return lexicon    


# modified
def retrofit(wordVecs, lexicon, numIters, alpha=1, beta=1):

  # The retrofitted vectors are initialized to be equal to the original vectors
  retrofittedVecs = deepcopy(wordVecs)

  for _ in range(numIters):
    for word in lexicon:
      if word in retrofittedVecs:
        neighbors = [neighbor for neighbor in lexicon[word] if neighbor in retrofittedVecs]
        numNeighbors = len(neighbors)
                
        if numNeighbors == 0:
          continue
                
        beta = 1 / numNeighbors

        # update (beta * numNeighbors will always be 1)
        retrofittedVecs[word] = sum([beta * retrofittedVecs[neighbor] for neighbor in neighbors]) +  alpha * wordVecs[word] / (beta * numNeighbors + alpha)
    
    return retrofittedVecs

# modified
if __name__=='__main__':

  if len(sys.argv) != 5:
    sys.exit("Usage: python modified.py input lexicon numiter output")

  wordVecs = read_word_vecs(sys.argv[1])  # input
  lexicon = read_lexicon(sys.argv[2])     # lexicon
  numIter = int(sys.argv[3])              # numiter
  outFileName = sys.argv[4]               # output
  
  
  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName) 

  # TODO: Read the wordnet and framenet word relations as a dictionary