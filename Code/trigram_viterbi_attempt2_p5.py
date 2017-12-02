import os, sys, math
import codecs
import copy
from module import read_in_file
import numpy as np


"""
PAAAAAAAAAAAAAAAAAAAAAART FIVEEEEEEEEEEEEEEEEEEEEEEE
Sentiment analysis based on emission parameters only
Saves file 'dev.p5.out'
:return: none
"""
def log(num):
    if num == 0 or 0.0:
        return -1000
    else:
        return math.log(num)

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']

def new_count(training_data, k):
    """
    Runs through entire training data to count the number of x,
    then runs through the training data again to modify x,
    count y, as well as counting the emissions
    :returns: emission counts, x counts, y counts
    """
    emission_count = {}
    transition_count = {}
    transition_ABC_count = {}
    x_count = {} # for #UNK# later
    y_count = {'START':len(training_data), 'STOP':len(training_data)}

    # Getting counts of x
    # time complexity len(training_data) ^ len(sentence)
    for sentence in training_data:
        for i in range(len(sentence)):
            curr_pair = sentence[i].split(' ')
            curr_x = curr_pair[0]

            if (len(curr_pair) > 2):
                for j in range(1, len(curr_pair) - 1):
                    curr_x += curr_pair[j]

            key = curr_x
            if (key not in x_count):
                x_count[key] = 1
            else:
                x_count[key] += 1

    # time complexity len(training_data) ^ len(sentence)
    for sentence in training_data:
        for i in range(len(sentence)):
            curr_pair = sentence[i].split(' ')
            curr_x = curr_pair[0]

            if (len(curr_pair) > 2):
                for j in range(1, len(curr_pair) - 1):
                    curr_x += curr_pair[j]
                curr_y = curr_pair[len(curr_pair) - 1]
            else:
                curr_y = curr_pair[1]

            # checking for unknowns
            if x_count[curr_x] < k:
                curr_x = "#UNK#"

            # counting emissions
            key = (curr_y, curr_x)
            if (key not in emission_count):
                emission_count[key] = 1
            else:
                emission_count[key] += 1

            # counting transition_params
            if (i == 0): # when there is nothing before you (first of sentence)
                key = ('START',curr_y)
            else:
                prev_pair = sentence[i-1].split(' ')
                prev_y = prev_pair[len(prev_pair) - 1]

                key = (prev_y, curr_y)
            if (key not in transition_count):
                transition_count[key] = 1
            else:
                transition_count[key] += 1

            if (i == len(sentence)-1): # when there is nothing behind you (last of sentence)
                key = (curr_y, 'STOP')
                if (key not in transition_count):
                    transition_count[key] = 1
                else:
                    transition_count[key] += 1

            # counting transition_ABC_count for transition_ABC_params
            if(i != len(sentence) - 1):
                next_pair = sentence[i+1].split(' ')
                next_y = next_pair[len(next_pair) - 1]

            if (i == 0): # when there is nothing before you (first of sentence)
                key = ('START',curr_y,next_y)
            else:
                prev_pair = sentence[i-1].split(' ')
                prev_y = prev_pair[len(prev_pair) - 1]

                key = (prev_y,curr_y,next_y)
            if(key not in transition_ABC_count):
                transition_ABC_count[key] = 1
            else:
                transition_ABC_count[key] += 1

            if (i == len(sentence)-2):  # when there is nothing behind you (last of sentence)
                key = (prev_y,curr_y,'STOP')
                if(key not in transition_ABC_count):
                    transition_ABC_count[key] = 1
                else:
                    transition_ABC_count[key] += 1

            # counting y
            key = curr_y
            if (key not in y_count):
                y_count[key] = 1
            else:
                y_count[key] += 1

    return emission_count,transition_count, transition_ABC_count, y_count, x_count

def emissions(x, y, emission_count, y_count):
    """
    Getting emission parameters of x and y
    Params emission_count and y_count taken from initial run of count()
    :returns: float probability
    """
    try:
        return float(emission_count[(y,x)])/float(y_count[y])

    except KeyError:
        return 0.0

def transitions(y1, y2, transition_count, y_count):
    """
    Getting transition parameters of y1 and y2
    Params transition_count and y_count taken from initial run of count()
    :returns: float probability
    """
    try:
        return float(transition_count[(y1,y2)])/float(y_count[y1])
    except KeyError:
        return 0.0

def transitions_ABC(y1, y2, y3, transition_ABC_count, transition_count):
    """
    Getting transition parameters of y1, y2 and y3
    Params transition_ABC_count and y_count taken from initial run of count()
    :returns: float probability
    """
    try:
        return float(transition_ABC_count[y1,y2,y3]/float(transition_count[y1,y2]))
    except KeyError:
        return 0.0


def get_parameters(emission_count, transition_count, transition_ABC_count, y_count):
    """
    Gets both the transition and emission parameters from the counts
    :returns: the transition and emission parameters as dictionaries
    """
    emission_params = {}
    for pair in emission_count:
        emission_params[pair] = emissions(pair[1], pair[0], emission_count, y_count)

    transition_params = {}
    for pair in transition_count:
        transition_params[pair] = transitions(pair[0], pair[1], transition_count, y_count)

    transition_ABC_params = {}
    for triplet in transition_ABC_count:
        transition_ABC_params[triplet] = transitions_ABC(triplet[0], triplet[1], triplet[2], transition_ABC_count, transition_count)

    return emission_params, transition_params, transition_ABC_params


def viterbi(x,a,b,c):
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

    for i in range(n+1):
        pi.append([])
        for j in range(T):
            pi[i].append([-1000,'O']) # idx 0 represents score, idx 1 represents parent node

    # Base case: start step
    for u in y:
        try:
            pi[0][u][0] = log(a[('START', states[u])]) + log(b[(states[u],x[0])])
        except KeyError:
            pi[0][u][0] = -1000
        pi[0][u][1] = 'START'

    # Recursive case
    for i in range(1,n):
        for u in y:
            for v in y:
                for t in y:
                    try:
                        p = (pi[i-1][v][0]) + log(0.9*a[(states[v], states[u])]) + log(b[(states[u], x[i])]) + log(0.1*c[(states[v], states[u], states[t])])
                    except KeyError:
                        p = -1000
                    if p >= pi[i][u][0]: # if it doesn't satisfy this condition for all nodes u, then the word would not be identified as an Entity
                        pi[i][u][0] = p
                        pi[i][u][1] = states[v]
                    # pi[i][u][v] = p

    # Second last case:
    for u in y:
        for v in y:
            try:
                p = (pi[n-2][v][0]) + log(0.9*a[(states[v], 'STOP')]) + log(0.1*c[(states[u], states[v], 'STOP')])
            except KeyError:
                p = -1000
            if p >= pi[n-1][0][0]:
                pi[n-1][0][0] = p
                pi[n-1][0][1] = states[v]

    # Base case: Final step
    for v in y:
        try:
            p = (pi[n-1][v][0]) + log(a[(states[v], 'STOP')])
        except KeyError:
            p = -1000
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
    emission_count, transition_count, transition_ABC_count, y_count, x_count = new_count(train_data, 3)
    print('done counting x, y, emissions')

    b, a, c = get_parameters(emission_count, transition_count, transition_ABC_count, y_count)
    print('done getting all transition and emission parameters')

    test_data = read_in_file(test_path)
    print('done reading test file')

    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, output_path)
    with codecs.open(os.path.join(save_path,'dev.p5_trigram.out'), 'w', 'utf-8') as file:
        for sentence in test_data:
            mod_sentence = []
            for word in sentence:
                # To check if word in test data appears in training data
                if word not in x_count or x_count[word] < 3:
                    mod_word = '#UNK#'
                else:
                    mod_word = word
                mod_sentence.append(mod_word)

            pi = viterbi(mod_sentence, a, b, c)
            output_states = back_propagation(pi)

            for i in range(len(sentence)):
                output = sentence[i] + ' ' + output_states[i+1] + '\n'
            # output = word + ' ' + optimum_y + '\n'
                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()

viterbi_sentiment_analysis('EN')
viterbi_sentiment_analysis('FR')
viterbi_sentiment_analysis('CN')
viterbi_sentiment_analysis('SG')

# trainFile = read_in_file('../Datasets/SG/train')
# emission_count, transition_count, y_count, x_count = count(trainFile, 3)
# emission_params, transition_params = get_parameters(emission_count, transition_count, y_count)
