# copy this:
# from module import read_in_file, count, emissions, transitions, get_parameters
import os
import sys

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

            # counting y
            key = curr_y
            if (key not in y_count):
                y_count[key] = 1
            else:
                y_count[key] += 1

    return emission_count,transition_count, y_count, x_count

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
    
def get_parameters(emission_count, transition_count, y_count):
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
    
    return emission_params, transition_params