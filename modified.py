import string
import gzip
import numpy as np
import sys
import os

from copy import deepcopy


def normalize(word):
  # words containing digits are replaced with '*NUM*'
  if any(char.isdigit() for char in word):
    return '*NUM*'
  # punctuations are replaced with '*PUNC*'
  elif word in string.punctuation:
    return '*PUNC*'
  # all other words are returned intact
  else:
    return word 


''' Read and normalize the embeddings '''
def read_embeddings(filename):
  print("\nReading embeddings...")
  # keys: words (string)
  # values: normalized vectors (NumPy array)
  embeds = {} 
  
  with (gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename, 'r')) as file:
    for line in file:
      elements = line.strip().split()
      word = elements[0]
      vector = np.array([float(value) for value in elements[1:]], dtype=float)
        
      # normalize vector (Euclidean norm)
      norm = np.linalg.norm(vector)
      embeds[word] = vector / norm  
  
  print("Reading embeddings done!")
  return embeds

  
''' Read the lexicon as a dictionary '''
def read_lexicon(filename):
  # keys: 1st word of each line (string)
  # values: all the other words in the line (list of strings)
  lexicon = {}  
  
  with open(filename, 'r') as file:
    for line in file:
      words = line.strip().split() 
      lexicon[normalize(words[0])] = [normalize(word) for word in words[1:]] 
  
  return lexicon    


''' Write retrofitted embeddings to specified file '''
def write_retrofitted_embeddings(embeddings, filename):
  print(f"\nWriting embeddings to {filename}")
  
  with open(filename, 'w') as file:
    for word, vector in embeddings.items():
        file.write(f"{word} ")
        np.savetxt(file, vector[np.newaxis], delimiter=' ', fmt='%.4f')    
  
  print("Writing embeddings done!\n")


''' Retrofit embeddings '''
def retrofit(originalEmbeds, lexicon, numIters, alpha=1, beta=1):

  # The retrofitted vectors are initialized to be equal to the original vectors
  retrofittedEmbeds = deepcopy(originalEmbeds)
  
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
          [beta * retrofittedEmbeds.get(neighbor.casefold(), 0.0) for neighbor in neighbors]) + alpha * originalEmbeds.get(word.casefold(), 0.0)) / (
          beta * numNeighbors + alpha)
    
  return retrofittedEmbeds


#########################################################################################################################################

if __name__=='__main__':

  '''Check for correct number of arguments'''
  if len(sys.argv) != 5:
    sys.exit("\nUsage: python modified.py inFile lexicon numIter outFile\n")
  

  '''Check for correct order of arguments'''
  # inFile
  if not (sys.argv[1].endswith('.gz') or sys.argv[1].endswith('.txt')):
    sys.exit("\nUsage: python modified.py inFile lexicon numIter outFile\ninFile should be a .gz or .txt file.\n")
  
  # lexicon
  if not (os.path.isfile(sys.argv[2]) or sys.argv[2].endswith('.txt')):
    sys.exit("\nUsage: python modified.py inFile lexicon numIter outFile\nlexicon should be a .txt file.\n")

  # numIter
  if not sys.argv[3].isdigit():
    sys.exit("\nUsage: python modified.py inFile lexicon numIter outFile\nnumIter should be a positive integer.\n")

  # outFile
  outDir = os.path.dirname(sys.argv[4])
  if outDir != '' and not os.path.isdir(outDir):
    sys.exit(f"Error: Output directory '{outDir}' does not exist.")

  if not sys.argv[4].endswith('.txt'):
    sys.exit("\nUsage: python modified.py inFile lexicon numIter outFile\noutFile should be a .txt file.\n")
  

  '''if all the checks pass, proceed'''
  embeddings = read_embeddings(sys.argv[1]) # inFile
  lexicon = read_lexicon(sys.argv[2])       # lexicon
  numIter = int(sys.argv[3])                # numIter
  outFileName = sys.argv[4]                 # outFile
  
  write_retrofitted_embeddings(retrofit(embeddings, lexicon, numIter), outFileName) 