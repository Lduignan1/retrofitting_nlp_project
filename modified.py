import string
import gzip
import numpy as np
import sys

from copy import deepcopy

# QUESTION: what would happen if we don't normalize the words?
def normalize(word):
  # words consisting entirely of digits (also '-' and '.') are replaced with '---num---'
  # QUESTION: do we also want words that contain digits to be replaced with '---num---'?
  if word.lstrip('-').replace('.', '').isdigit():
    return '---num---'
  # punctuations are replaced with '---punc---'
  elif word in string.punctuation:
    return '---punc---'
  # all other words are converted to lowercase
  else:
    return word # removed .lower()


''' Read and normalize the embeddings '''
def read_embeddings(filename):
  # keys: words (string)
  # values: normalized vectors (NumPy array)
  embeds = {} 
  
  with (gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename, 'r')) as file:
    for line in file:
      elements = line.strip().lower().split()
      word = elements[0]
      vector = np.array([float(value) for value in elements[1:]], dtype=float)
      
      # normalize vector (Euclidean norm)
      norm = np.linalg.norm(vector)
      embeds[word] = vector / norm  
  
  return embeds

  
''' Read the lexicon as a dictionary '''
def read_lexicon(filename):
  # keys: 1st word of each line (string)
  # values: all the other words in the line (list of strings)
  lexicon = {}  
  
  with open(filename, 'r') as file:
    for line in file:
      words = line.strip().split()  # removed .lower() after line
      lexicon[normalize(words[0])] = [normalize(word) for word in words[1:]] 
  
  return lexicon    


''' Write retrofitted embeddings to specified file '''
def write_retrofitted_embeddings(embeddings, filename):
  with open(filename, 'w') as file:
    for word, vector in embeddings.items():
        file.write(f"{word} ")
        np.savetxt(file, vector[np.newaxis], delimiter=' ', fmt='%.4f')    


''' Retrofit embeddings '''
def retrofit(embeddings, lexicon, numIters, alpha=1, beta=1):

  # The retrofitted vectors are initialized to be equal to the original vectors
  retrofittedEmbeds = deepcopy(embeddings)
  
  # creating a separate set to store the lowercase versions of the keys (words) in retrofittedEmbeds 
  # this allows us to perform case-insensitive key comparison while retaining the original case of the words
  embedKeys = {key.lower() for key in retrofittedEmbeds.keys()}

  for _ in range(numIters):
    for word in lexicon:
      # we only care about words that are both in the lexicon and in the embeddings
      if word.lower() in embedKeys:
        
        # neighbors is a list of all the words that are similar to the current word (based on the given lexicon)
        neighbors = [neighbor for neighbor in lexicon[word] if neighbor.lower() in embedKeys]
        
        numNeighbors = len(neighbors)
                
        if numNeighbors == 0:
          continue
                
        beta = 1 / numNeighbors

        # update step
        # (beta * numNeighbors will always be 1)
        # return a default value of 0.0 if a key is not found
        retrofittedEmbeds[word] = (sum(
          [beta * retrofittedEmbeds.get(neighbor.casefold(), 0.0) for neighbor in neighbors]) + alpha * embeddings.get(word.casefold(), 0.0)) / (
          beta * numNeighbors + alpha)
    
  return retrofittedEmbeds


#########################################################################################################################################

if __name__=='__main__':

  # ISSUE: should still check for the correct order of the arguments
  if len(sys.argv) != 5:
    sys.exit("Usage: python modified.py input lexicon numiter output")

  embeddings = read_embeddings(sys.argv[1]) # input
  lexicon = read_lexicon(sys.argv[2])       # lexicon
  numIter = int(sys.argv[3])                # numiter
  outFileName = sys.argv[4]                 # output
  
  write_retrofitted_embeddings(retrofit(embeddings, lexicon, numIter), outFileName) 