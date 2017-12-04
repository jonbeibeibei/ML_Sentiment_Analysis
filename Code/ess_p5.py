import os, sys, math
import codecs
import copy
from module import read_in_file, emissions, transitions, get_parameters
from viterbi_p3 import viterbi
from maxmarginal_p4 import maximum_marginal_sentence

"""
Part 5
Entity & Sentiment Separation Analysis
Entity and Sentiment Prediction done via separation
Saves file 'dev.p5_ess.out'
:return: none
"""

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']
sentiment_only_states = ['O', 'positive', 'negative', 'neutral']
entity_only_states = ['O', 'B-', 'I-']

def count_sentiment_only(training_data, k):
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

            if 'positive' in curr_y:
                curr_y = 'positive'

            if 'negative' in curr_y:
                curr_y = 'negative'

            if 'neutral' in curr_y:
                curr_y = 'neutral'

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

                if 'positive' in prev_y:
                    prev_y = 'positive'

                if 'negative' in prev_y:
                    prev_y = 'negative'

                if 'neutral' in prev_y:
                    prev_y = 'neutral'

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

def count_entity_only(training_data, k):
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

            if 'B-' in curr_y:
                curr_y = 'B-'

            if 'I-' in curr_y:
                curr_y = 'I-'

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

                if 'B-' in prev_y:
                    prev_y = 'B-'

                if 'I-' in prev_y:
                    prev_y = 'I-'

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

def viterbi_sentiment_only(x,a,b):
    """
    For Viterbi that only considers sentiments
    :params x: list -- sequence of modified words/observations
    :params a: transition parameters from training set
    :params b: emission parameters from training set

    Executes the Viterbi algorithm for each tweet
    :returns: pi, a matrix that contains the best score and parent node of each node
    """
    # Initializing the pi matrix with 0s
    y = [0,1,2,3] # 'O', 'positive', 'negative', 'neutral'
    pi = []
    T = len(y)
    n = len(x)

    for i in range(n+1):
        pi.append([])
        for j in range(T):
            pi[i].append([0,'O']) # idx 0 represents score, idx 1 represents parent node

    # Base case: start step
    for u in y:
        try:
            pi[0][u][0] = (a[('START', sentiment_only_states[u])]) * (b[(sentiment_only_states[u],x[0])])
        except KeyError:
            pi[0][u][0] = 0.0
        pi[0][u][1] = 'START'

    # Recursive case
    for i in range(1,n):
        for u in y:
            for v in y:
                try:
                    p = (pi[i-1][v][0]) * (a[(sentiment_only_states[v], sentiment_only_states[u])]) * (b[(sentiment_only_states[u], x[i])])
                except KeyError:
                    p = 0.0
                if p >= pi[i][u][0]: # if it doesn't satisfy this condition for all nodes u, then the word would not be identified as an Entity
                    pi[i][u][0] = p
                    pi[i][u][1] = sentiment_only_states[v]

    # Base case: Final step
    for v in y:
        try:
            p = (pi[n-1][v][0]) * (a[(sentiment_only_states[v], 'STOP')])
        except KeyError:
            p = 0.0
        if p >= pi[n][0][0]:
            pi[n][0][0] = p
            pi[n][0][1] = sentiment_only_states[v]

    return pi

def back_propagation_sentiment_only(pi):
    """
    for Viterbi that only considers sentiments
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
    state_index = sentiment_only_states.index(state)

    for i in range(len(pi)-2,0,-1):
        state = pi[i][state_index][1]
        labels[i] = state
        state_index = sentiment_only_states.index(state)

    labels[0] = 'START'

    return labels

def viterbi_entity_only(x,a,b):
    """
    for Viterbi that only considers entities
    :params x: list -- sequence of modified words/observations
    :params a: transition parameters from training set
    :params b: emission parameters from training set

    Executes the Viterbi algorithm for each tweet
    :returns: pi, a matrix that contains the best score and parent node of each node
    """
    # Initializing the pi matrix with 0s
    y = [0,1,2] # 'O', 'B-', 'I-'
    pi = []
    T = len(y)
    n = len(x)

    for i in range(n+1):
        pi.append([])
        for j in range(T):
            pi[i].append([0,'O']) # idx 0 represents score, idx 1 represents parent node

    # Base case: start step
    for u in y:
        try:
            pi[0][u][0] = (a[('START', entity_only_states[u])]) * (b[(entity_only_states[u],x[0])])
        except KeyError:
            pi[0][u][0] = 0.0
        pi[0][u][1] = 'START'

    # Recursive case
    for i in range(1,n):
        for u in y:
            for v in y:
                try:
                    p = (pi[i-1][v][0]) * (a[(entity_only_states[v], entity_only_states[u])]) * (b[(entity_only_states[u], x[i])])
                except KeyError:
                    p = 0.0
                if p >= pi[i][u][0]: # if it doesn't satisfy this condition for all nodes u, then the word would not be identified as an Entity
                    pi[i][u][0] = p
                    pi[i][u][1] = entity_only_states[v]

    # Base case: Final step
    for v in y:
        try:
            p = (pi[n-1][v][0]) * (a[(entity_only_states[v], 'STOP')])
        except KeyError:
            p = 0.0
        if p >= pi[n][0][0]:
            pi[n][0][0] = p
            pi[n][0][1] = entity_only_states[v]

    return pi

def back_propagation_entity_only(pi):
    """
    For Viterbi that only considers entities
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
    state_index = entity_only_states.index(state)

    for i in range(len(pi)-2,0,-1):
        state = pi[i][state_index][1]
        labels[i] = state
        state_index = entity_only_states.index(state)

    labels[0] = 'START'

    return labels




def ess_analysis(language):
    """
    Performs ESS algorithm on our test data
    Params: Language of dataset you wish to run the analysis on
    :returns: None, writes to output file
    """
    k = 3

    training_path = '../Datasets/' + language +  '/train'
    test_path = '../Datasets/' + language + '/dev.in'
    output_path = '../EvalScript/' + language

    optimal_y_dict = {}

    train_data = read_in_file(training_path)
    print('done reading training file')
    s_emission_count, s_transition_count, s_y_count, s_x_count = count_sentiment_only(train_data, k)
    print('done counting x, y, emissions for sentiment only')

    s_b, s_a = get_parameters(s_emission_count, s_transition_count, s_y_count)
    print('done getting all transition and emission parameters for sentiment only')

    e_emission_count, e_transition_count, e_y_count, e_x_count = count_entity_only(train_data, k)
    print('done counting x, y, emissions for entity only')

    e_b, e_a = get_parameters(e_emission_count, e_transition_count, e_y_count)
    print('done getting all transition and emission parameters for entity only')


    test_data = read_in_file(test_path)
    print('done reading test file')
    #
    main_path = os.path.dirname(__file__)
    save_path = os.path.join(main_path, output_path)
    with codecs.open(os.path.join(save_path,'dev.p5_ess.out'), 'w', 'utf-8') as file:
        for sentence in test_data:
            mod_sentence = []
            for word in sentence:
                # To check if word in test data appears in training data
                if word not in s_x_count or s_x_count[word] < k:
                    mod_word = '#UNK#'
                else:
                    mod_word = word
                mod_sentence.append(mod_word)

            # Run viterbi but only to get the sentiments
            sentiment_pi = viterbi_sentiment_only(mod_sentence, s_a, s_b)
            output_states_sentiment = back_propagation_sentiment_only(sentiment_pi)

            #Run viterbi but only to get the entities
            entity_pi = viterbi_entity_only(mod_sentence, e_a, e_b)
            output_states_entity = back_propagation_entity_only(entity_pi)

            # print('sentiment: ', output_states_sentiment)
            # print('entity: ', output_states_entity)

            fixed_output_states = output_states_sentiment

            # Compare output states from the viterbi_entity_only and viterbi_sentiment_only
            for i in range(len(sentence)):
                entity_label = output_states_entity[i+1]
                sentiment_label = output_states_sentiment[i+1]

                if(entity_label != 'O'):
                    if(sentiment_label != 'O'):
                        fixed_output_states[i+1] = entity_label + sentiment_label

                    else:
                        fixed_output_states[i+1] = 'O'



            for j in range(len(sentence)):
                curr_state = fixed_output_states[j+1]
                #Check if all previous entries in the state sequence are Os
                if(curr_state != 'O'):
                    for check in range(1,i):
                        if(fixed_output_states[check] != 'O'):
                            flag = False # If not all Os before, then set flag to False
                            break
                        flag = True #If all Os before, set flag to True

                    if(flag == True): #If flag == True, check if the first entity encountered is B-, if not make sure it is B-
                        if('I-' in curr_state):
                            curr_state.replace('I-','B-')
                        fixed_output_states[j+1] = curr_state





            for i in range(len(sentence)):
                output = sentence[i] + ' ' + fixed_output_states[i+1] + '\n'
            # output = word + ' ' + optimum_y + '\n'
                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()

if __name__ == '__main__':
    ess_analysis('EN')
    ess_analysis('FR')
    # ess_analysis('CN')
    # ess_analysis('SG')
