import os
import math
import codecs
import copy
from module import read_in_file, count, emissions, transitions, get_parameters

"""
Part 2
Sentiment analysis based on emission parameters only
Saves file 'dev.p2.out'
:return: none
"""

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']

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
    output_path = '../EvalScript/' + language

    optimal_y_dict = {}

    train_data = read_in_file(training_path)
    print('done reading training file')
    emission_count, transition_count, y_count, x_count = count(train_data, 3)
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
