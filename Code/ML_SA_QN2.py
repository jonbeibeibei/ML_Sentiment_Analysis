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

def emissions(x, y, training_data):

    emission_count = 0
    # x_count = 0
    y_count = 0

    for sentence in training_data:
        for i in range(len(sentence)):
            curr_sentence = sentence[i].split(' ')
            curr_x = curr_sentence[0]
            curr_y = curr_sentence[1]
            # if curr_x == x:
            #     x_count += 1
            if curr_y == y:
                y_count += 1
                if curr_x == x:
                    emission_count += 1

    # if x_count == 0:
    #     return float(1/y_count)

    return float(emission_count/y_count)

def get_optimal_y(x, training_data):
    optimum_y_prob = 0
    optimum_y = ''
    for state in states:
        y_prob = emissions(x,state,training_data)
        if y_prob > optimum_y_prob:
            optimum_y_prob = y_prob
            optimum_y = state

    return optimum_y

def simple_sentiment_analysis(training_path, test_path,output_path):

    optimal_y_dict = {}

    trainFile = read_in_file(training_path)
    train_words, modified_training_data = modify_train_data(3,trainFile)
    print('done modifying training_data')

    testFile = read_in_file(test_path)
    modified_test_data = modify_test_data(train_words, testFile)
    print('done modifying test_data')

    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, output_path)
    with codecs.open(os.path.join(save_path,'dev.p2.out'), 'w', 'utf-8') as file:
        for sentence in range(len(modified_test_data)):
            for word in range(len(modified_test_data[sentence])):
                if modified_test_data[sentence][word] in optimal_y_dict.keys():
                    optimum_y = optimal_y_dict[modified_test_data[sentence][word]]
                else:
                    optimum_y = get_optimal_y(modified_test_data[sentence][word], modified_training_data)
                    optimal_y_dict[modified_test_data[sentence][word]] = optimum_y
                output = testFile[sentence][word] + ' ' + optimum_y + '\n'
                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()





def save_file(self, result):
    """
    Saves given input into 'dev.p2.out' file on given path
    :returns: none
    """
    output = ""
    for tweet in result:
        for pair in tweet:
            output += pair[0] + " " + pair[1] + "\n"
        output += "\n"

    path = '../Datasets/Demo'
    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, path)
    with codecs.open(os.path.join(save_path,'dev.p2.out'), 'w', 'utf-8') as file:
        file.write(output)


def modify_train_data(k, training_data):
    """
    Replace words that appear less than k times in training set with #UNK#
    Appends word to a list for test data modification
    :returns: none
    """
    #Get dictionary of all the words contained in training set
    word_dict = {}
    for sentence in training_data:
        for i in range(len(sentence)):
            curr_sentence = sentence[i].split(' ')
            curr_x = curr_sentence[0]
            if curr_x not in word_dict:
                word_dict[curr_x] = 0
            else:
                word_dict[curr_x] += 1

    # print(word_dict)
    # Clone the training data
    print('copying the train_data out...')
    modified_training_data = copy.deepcopy(training_data)
    output_word_dict = word_dict.copy()
    print('copied!')
    #Iterate over all words and check values, if < k, replace with "#UNK#"
    for key, value in word_dict.items():

        if value < k:
            for sentence in modified_training_data:
                for i in range(len(sentence)):
                    curr_sentence = sentence[i].split(' ')
                    curr_x = curr_sentence[0]
                    curr_y = curr_sentence[1]
                    if key == curr_x:
                        sentence[i] = '#UNK#' + ' ' + curr_y
                        output_word_dict.pop(key,None)

    return output_word_dict, modified_training_data

def modify_test_data(train_words, test_data):
    """
    Replace words that do not appear in train_words
    Appends word to a list for test data modification
    :returns: none
    """

    print('copying the test_data out...')
    modified_test_data = copy.deepcopy(test_data)
    print('copied!')

    for sentence in modified_test_data:
        for i in range(len(sentence)):
            if sentence[i] not in train_words.keys():
                sentence[i] = '#UNK#'


    return modified_test_data

simple_sentiment_analysis('../Datasets/SG/train','../Datasets/SG/dev.in','../Datasets/SG')
