import numpy as np

"""
TODO: 
1) extract semanticaly related words from existing sources (WordNet, FrameNet etc)
2) implement algo
3) implement evaluation metrics

"""
# are we supposed to recreate everything before the retrofit function too?

# ALGO:
# 1: train the word vectors independent of the information in the semantic lexicons

# 2: retrofit (update)
def retrofit(wordVecs, lexicon, numIters, alpha, beta):
    """
    Q := Q ̂  // The vectors in Q are initialized to be equal to the vectors in Q ̂
    Repeat for each vector until (stopping criteria):
        take the first derivative of Ψ with respect to one qi vector: Ψ'(q_i)

        update q_i

    """
    vocab = set(wordVecs.keys())
    
    # The retrofitted vectors are initialized to be equal to the original vectors
    retrofittedVecs = wordVecs.copy()
    
    for _ in range(numIters):
        for word in lexicon:
            if word in retrofittedVecs:
                neighbors = [neighbor for neighbor in lexicon[word] if neighbor in retrofittedVecs]
                numNeighbors = len(neighbors)
                
                if numNeighbors == 0:
                    continue
                
                # update
                retrofittedVecs[word] = np.sum([beta * retrofittedVecs[neighbor] for neighbor in neighbors]) +  alpha * retrofittedVecs[word] / (beta * numNeighbors + alpha)
    
    return retrofittedVecs


# EVALUATION