import os
import sys
import math
import codecs
import copy


"""
PAAAAAAAAAAAAAAAAAAAAAART TWOOOOOOOOOOOOOOOOOOOOOOOOO
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
    transition_count = {}
    x_count = {} # for #UNK# later
    y_count = {}

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

def simple_sentiment_analysis(language):
    """
    Performs simple sentiment analysis only using emission parameters.
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