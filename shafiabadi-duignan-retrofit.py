import gzip
import numpy as np
import sys
import os
from lexicon import Lexicon

from copy import deepcopy


''' Read and normalize the embeddings '''
def read_embeddings(filename):
  print("\nReading embeddings...")
  # keys: words (string)
  # values: normalized vectors (NumPy array)
  embeds = {} 
  
  # using encoding='utf-8' to avoid UnicodeEncodeError on some systems
  with (gzip.open(filename, 'rt', encoding='utf-8') if filename.endswith('.gz') else open(filename, 'r', encoding='utf-8')) as file:
    for line in file:
      elements = line.strip().split()
      word = elements[0]
      vector = np.array([float(value) for value in elements[1:]], dtype=float)
        
      # normalize vector (vector / Euclidean norm)
      norm = np.linalg.norm(vector)
      embeds[word] = vector / norm  
  
  print("Reading embeddings done!")
  return embeds


''' Write retrofitted embeddings to specified file '''
def write_retrofitted_embeddings(embeddings, filename):
  print(f"\nWriting embeddings to {filename}")
  
  with open(filename, 'w', encoding='utf-8') as file:
    for word, vector in embeddings.items():
        file.write(f"{word} ")
        np.savetxt(file, vector[np.newaxis], delimiter=' ', fmt='%.4f')    
  
  print("Writing embeddings done!\n")


''' Retrofit embeddings '''
def retrofit(originalEmbeds, lexicon, numIters, alpha=1, beta=1):
  print("\nRetrofitting word embeddings...")

  # The retrofitted vectors are initialized to be equal to the original vectors
  retrofittedEmbeds = deepcopy(originalEmbeds)

  for _ in range(numIters):
    for word in lexicon:
      # we only care about words that are both in the lexicon and in the embeddings
      if word in retrofittedEmbeds:
        
        # neighbors is a list of all the words that are similar to the current word (based on the given lexicon)
        neighbors = [neighbor for neighbor in lexicon[word] if neighbor in retrofittedEmbeds]
        
        numNeighbors = len(neighbors)
                
        if numNeighbors == 0:
          continue
        
        beta = 1 / numNeighbors

        # update step
        # (beta * numNeighbors will always be 1)
        retrofittedEmbeds[word] = (sum(
          [beta * retrofittedEmbeds[neighbor] for neighbor in neighbors]) + alpha * originalEmbeds[word]) / (
          beta * numNeighbors + alpha)

  print("Retrofitting done!")  
  return retrofittedEmbeds


#########################################################################################################################################

''' Display the online help '''
def display_help():
    with open('online_help.txt', 'r') as file:
        help_content = file.read()
        print(help_content)


if __name__=='__main__':

  '''Check for correct number of arguments'''
  
  if len(sys.argv) == 2:
    # Check if the help option is provided
    if '--help' in sys.argv or '-h' in sys.argv:
        display_help()
        sys.exit()
    else:
      print("\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>")
      print("\nTo access the online help: python shafiabadi-duignan-retrofit.py --help\n")
      sys.exit(1)
  
  if len(sys.argv) != 6:
    print("\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>")
    print("\nTo access the online help: python shafiabadi-duignan-retrofit.py --help\n")
    sys.exit(1)
  

  '''Check for correct order and format of arguments'''
  
  # language
  language = sys.argv[2].lower()
  if not (language in ['eng', 'fra']):
    sys.exit(
      "\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>\nPossible languages: eng or fra (case insensitive)\n")

  # lexicon
  lex = sys.argv[3].lower()
  if not (lex in ['wn', 'wn+', 'wordnet', 'wordnet+'] or lex == 'ppdb'):
    sys.exit("\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>\nPossible lexicons: PPDB or WordNet(WN) or WordNet+(WN+) (case insensitive)\n")

  # numIter
  if not sys.argv[4].isdigit():
    sys.exit("\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>\nIterations should be a positive integer.\n")

  # outFile
  outDir = os.path.dirname(sys.argv[5])
  if outDir != '' and not os.path.isdir(outDir):
    sys.exit(f"\nError: Output directory '{outDir}' does not exist.\n")

  if not sys.argv[5].endswith('.txt'):
    sys.exit("\nUsage: python shafiabadi-duignan-retrofit.py <embeddings_file_path> <language> <lexicon> <iterations> <output_file_path>\nOutput file should be a .txt file.\n")
  

  '''if all the checks pass, proceed'''
  embeddings = read_embeddings(sys.argv[1]) # inFile
  numIter = int(sys.argv[4])                # numIter
  outFileName = sys.argv[5]                 # outFile
  
  # lexicon
  lexicon = Lexicon(lang=language)                       
  if lex == 'ppdb':
    lexicon = lexicon.read_ppdb()
  elif lex in ['wn', 'wordnet']:
    lexicon = lexicon.wn_synonyms()
  elif lex in ['wn+', 'wordnet+']:
    lexicon = lexicon.wn_all_relations()
  
  write_retrofitted_embeddings(retrofit(embeddings, lexicon, numIter), outFileName) 