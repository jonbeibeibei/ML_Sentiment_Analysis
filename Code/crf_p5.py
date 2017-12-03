import os, sys, math
import codecs
import copy
from module import read_in_file, count, emissions, transitions, get_parameters
from itertools import combinations_with_replacement as cr
import numpy as np
from viterbi_p3 import viterbi


"""
PAAAAAAAAAAAAAAAAAAAAAART FIVEEEEEEEEEEEEEEEEEEEEEEE
Sentiment analysis based on emission parameters only
Saves file 'dev.p5.out'
:return: none
"""
def log(num):
    if num == 0 or 0.0:
        return -sys.maxsize
    else:
        return math.log(num)

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']


def viterbi_CRF(x,a,b):
    """
    :params x: list -- sequence of modified words/observations
    :params a: transition parameters from training set
    :params b: emission parameters from training set

    Executes the Viterbi algorithm for each tweet
    :returns: pi, a matrix that contains the best score and parent node of each node
    """
    # Initializing the pi matrix with 0s
    y = [0,1,2,3,4,5,6] # corresponds to 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O'
    pi = []
    T = len(y)
    n = len(x)

    Zx = 0.0
    p = []
    print('len', n)

    for item in cr(states, n):
        curr_state = item[0]
        try:
            p.append(a[('START', curr_state)] * b[(curr_state,x[0])])
        except KeyError:
            p.append(0.0)
        Zx += math.exp(p[0])

        for j in range(1,n):
            curr_state = item[j]
            prev_state = item[j-1]
            try:
                p.append(a[(prev_state, curr_state)] * b[(curr_state,x[0])] * p[j-1])
            except KeyError:
                p.append(0.0)
            Zx += math.exp(p[j])

        curr_state = item[n-1]
        try:
            p.append(a[(curr_state, 'STOP')] * p[n-1])
        except KeyError:
            p.append(0.0)
        Zx += math.exp(p[n])


    for i in range(n+1):
        pi.append([])
        for j in range(T):
            pi[i].append([0,'O']) # idx 0 represents score, idx 1 represents parent node

    # Base case: start step
    for u in y:
        try:
            fx = (a[('START', states[u])]) * (b[(states[u],x[0])])
            pi[0][u][0] = math.exp(fx) / Zx
        except KeyError:
            pi[0][u][0] = 0.0
        pi[0][u][1] = 'START'

    # Recursive case
    for i in range(1,n):
        for u in y:
            for v in y:
                try:
                    fx = (pi[i-1][v][0]) * (a[(states[v], states[u])]) * (b[(states[u], x[i])])
                    p = math.exp(fx) / Zx
                except KeyError:
                    p = 0.0
                if p >= pi[i][u][0]: # if it doesn't satisfy this condition for all nodes u, then the word would not be identified as an Entity
                    pi[i][u][0] = p
                    pi[i][u][1] = states[v]

    # Base case: Final step
    for v in y:
        try:
            fx = (pi[n-1][v][0]) * (a[(states[v], 'STOP')])
            p = math.exp(fx) / Zx
        except KeyError:
            p = 0.0
        if p >= pi[n][0][0]:
            pi[n][0][0] = p
            pi[n][0][1] = states[v]

    return pi

def back_propagation(pi):
    """
    Takes in the pi matrix from viterbi() and back propagates to get each best parent node
    :returns: a list of labels (does not include start and stop)
    """
    labels = []

    for i in range(len(pi)):
        labels.append(0)

    # print(tweet_optimal_y_sequence)
    #Backpropagate to get Optimal State Sequence for Tweet
    state = pi[len(pi)-1][0][1]
    labels[len(pi)-1] = state
    state_index = states.index(state)

    for i in range(len(pi)-2,0,-1):
        state = pi[i][state_index][1]
        labels[i] = state
        state_index = states.index(state)

    labels[0] = 'START'

    return labels

def viterbi_sentiment_analysis(language):
    """
    Performs viterbi algorithm on our test data
    Params: Language of dataset you wish to run the analysis on
    :returns: None, writes to output file
    """

    training_path = '../Datasets/' + language +  '/train'
    test_path = '../Datasets/' + language + '/dev.in'
    output_path = '../EvalScript/' + language

    optimal_y_dict = {}

    train_data = read_in_file(training_path)
    print('done reading training file')
    emission_count, transition_count, y_count, x_count = count(train_data, 3)
    print('done counting x, y, emissions')

    b, a = get_parameters(emission_count, transition_count, y_count)
    print('done getting all transition and emission parameters')

    test_data = read_in_file(test_path)
    print('done reading test file')

    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, output_path)
    with codecs.open(os.path.join(save_path,'dev.p5.out'), 'w', 'utf-8') as file:
        for sentence in test_data:
            mod_sentence = []
            for word in sentence:
                # To check if word in test data appears in training data
                if word not in x_count or x_count[word] < 3:
                    mod_word = '#UNK#'
                else:
                    mod_word = word
                mod_sentence.append(mod_word)

            if(len(mod_sentence) < 20):
                pi = viterbi_CRF(mod_sentence, a, b)
                output_states = back_propagation(pi)

            else:
                pi = viterbi(mod_sentence, a, b)
                output_states = back_propagation(pi)

            for i in range(len(sentence)):
                output = sentence[i] + ' ' + output_states[i+1] + '\n'
            # output = word + ' ' + optimum_y + '\n'
                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()

viterbi_sentiment_analysis('EN')
# viterbi_sentiment_analysis('FR')
# viterbi_sentiment_analysis('CN')
# viterbi_sentiment_analysis('SG')

# trainFile = read_in_file('../Datasets/SG/train')
# emission_count, transition_count, y_count, x_count = count(trainFile, 3)
# emission_params, transition_params = get_parameters(emission_count, transition_count, y_count)
