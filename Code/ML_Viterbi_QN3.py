import os
import sys
import math
import codecs
import copy


"""
PAAAAAAAAAAAAAAAAAAAAAART THREEEEEEEEEEEEEEEEEEEEEEE
Sentiment analysis based on emission parameters only
Saves file 'dev.p2.out'
:return: none
"""

def read_in_file(path):
    """
    Reads in test/training file
    :returns: data_array
    """
    main_path = os.path.dirname(__file__)
    input_path = os.path.join(main_path, path)
    output = []
    inputFile = open(input_path, 'r', encoding='utf-8').read()

    for i in inputFile.split('\n\n'):
        tweet = []
        for j in i.split('\n'):
            if j != '':
                tweet.append(j)

        if len(tweet) > 0:
            output.append(tweet)

    return output
    inputFile.close()

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']

def count(training_data, k):
    """
    Runs through entire training data to count the number of x, 
    then runs through the training data again to modify x, 
    count y, as well as counting the emissions
    :returns: emission counts, x counts, y counts
    """
    emission_count = {}
    x_count = {} # for #UNK# later
    y_count = {}
    
    # Getting counts of x
    # time complexity len(training_data) ^ len(sentence)
    for sentence in training_data:
        for i in range(len(sentence)):
            curr_sentence = sentence[i].split(' ')
            curr_x = curr_sentence[0]
            
            key = curr_x
            if (key not in x_count):
                x_count[key] = 1
            else:
                x_count[key] += 1
            
    # time complexity len(training_data) ^ len(sentence)
    for sentence in training_data:
        for i in range(len(sentence)):
            curr_sentence = sentence[i].split(' ')
            curr_x = curr_sentence[0]
            curr_y = curr_sentence[1]
            
            # checking for unknowns
            if x_count[curr_x] < k:
                curr_x = "#UNK#"
            
            # counting emissions    
            key = (curr_y, curr_x)
            if (key not in emission_count):
                emission_count[key] = 1
            else:
                emission_count[key] += 1
                
            # counting y
            key = curr_y
            if (key not in y_count):
                y_count[key] = 1
            else:
                y_count[key] += 1
            
    return emission_count, y_count, x_count

def emissions(x, y, emission_count, y_count):
    """
    Getting emission parameters of x and y
    Params emission_count and y_count taken from initial run of count()
    :returns: float probability
    """
    try:
        return float(emission_count[(y,x)]/y_count[y])
    
    except KeyError:
        return 0.0

def get_optimal_y(x, emission_count, y_count):
    optimum_y_prob = 0
    optimum_y = ''
    for state in states:
        y_prob = emissions(x,state,emission_count,y_count)
        if y_prob >= optimum_y_prob:
            optimum_y_prob = y_prob
            optimum_y = state

    return optimum_y

def viterbi(self,x,y,a,b):
    """
    :params x: list -- sequence of modified words
    :params y: list -- integers that corresponds to the index of self.states
    :params a: transition parameters from training set
    :params b: emission parameters from training set

    Executes the Viterbi algorithm for each tweet
    :returns: pi, a matrix that contains the best score and parent node of each node
    """
    # Initializing the pi matrix with 0s
    pi = []
    T = len(y)
    n = len(x)
    for i in range(n+1):
        pi.append([])
        for j in range(T):
            pi[i].append([0,0]) # idx 0 represents score, idx 1 represents parent node

    # Base case: start step
    for u in y:
        try:
            pi[0][u][0] = (a[('START', self.states[u])]) + (b[(x[0],self.states[u])])
        except KeyError:
            pi[0][u][0] = 0.0
        pi[0][u][1] = 'START'

    # Recursive case
    for i in range(1,n):
        for u in y:
            for v in y:
                try:
                    p = (pi[i-1][v][0]) + (a[(self.states[v], self.states[u])]) + (b[(x[i], self.states[u])])
                except KeyError:
                    p = 0.0
                if p >= pi[i][u][0]:
                    pi[i][u][0] = p
                    pi[i][u][1] = self.states[v]

    # Base case: Final step
    for v in y:
        try:
            p = (pi[n-1][v][0]) + log(a[(self.states[v], 'STOP')])
        except KeyError:
            p = 0.0
        if p >= pi[n][0][0]:
            pi[n][0][0] = p
            pi[n][0][1] = self.states[v]
    # print(pi)
    return pi

def viterbi_sentiment_analysis(language):
    """
    Performs viterbi algorithm on our test data
    Params: Language of dataset you wish to run the analysis on
    :returns: None, writes to output file 
    """
    
    training_path = '../Datasets/' + language +  '/train'
    test_path = '../Datasets/' + language + '/dev.in'
    output_path = '../Datasets/' + language

    optimal_y_dict = {}

    train_data = read_in_file(training_path)
    print('done reading training file')
    emission_count, y_count, x_count = count(train_data, 3)
    print('done counting x, y, emissions')

    test_data = read_in_file(test_path)
    print('done reading test file')

    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, output_path)
    with codecs.open(os.path.join(save_path,'dev.p2.out'), 'w', 'utf-8') as file:
        for sentence in test_data:
            for word in sentence:
                # To check if word in test data appears in training data
                if word not in x_count:
                    mod_word = '#UNK#'
                else:
                    mod_word = word
                # To calculate best label
                if mod_word in optimal_y_dict:
                    optimum_y = optimal_y_dict[mod_word]
                else:
                    optimum_y = get_optimal_y(mod_word, emission_count, y_count)
                    optimal_y_dict[mod_word] = optimum_y
                output = word + ' ' + optimum_y + '\n'
                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()


simple_sentiment_analysis('SG')
simple_sentiment_analysis('EN')
simple_sentiment_analysis('FR')
simple_sentiment_analysis('CN')

# trainFile = read_in_file('../Datasets/SG/train')
# emission_count, y_count, x_count = count(trainFile, 3)
# print(x_count)