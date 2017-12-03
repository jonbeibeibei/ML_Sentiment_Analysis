import os
import math
import codecs
from module import read_in_file, count, emissions, transitions, get_parameters

"""
PAAAAAAAAAAAAAAAAAAAAAART FOURRRRRRRRRRRRRRRRRRRRRRR
Sentiment analysis based on emission parameters only
Saves file 'dev.p4.out'
:return: none
"""

states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']
# states = ['O','B-positive', 'B-negative', 'I-positive', 'I-negative','I-neutral', 'B-neutral']
# states = ['O','B-neutral','B-positive', 'B-negative', 'I-positive', 'I-negative','I-neutral']

def log(num):
    if num == 0 or 0.0:
        return -sys.maxsize
    else:
        return math.log(num)

def maximum_marginal_sentence(mod_sentence, a, b):
    output_states = []


    # Initializing the pi matrix with 0s
    y = [0,1,2,3,4,5,6] # corresponds to 'B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O'
    alpha = []
    beta = []
    T = len(y)
    n = len(mod_sentence)
    for i in range(n):
        alpha.append([])
        beta.append([])
        output_states.append(0)
        for j in range(T):
            alpha[i].append(0) # idx 0 represents score, idx 1 represents parent node
            beta[i].append(0)


    #Populating alpha and beta values

    #Base case:
    for u in y:
        try:
            alpha[0][u] = (a['START',states[u]])
            beta[n-1][u] = (a[states[u],'STOP']) * (b[states[u],mod_sentence[n-1]])

        except KeyError:
            alpha[0][u] = 0
            # beta[n-1][u] = 0

    #Recursive cases:
        #Alpha values
    for i in range(1,n):
        for u in y:
            for v in y:
                try:
                    alpha[i][u] += alpha[i-1][v] * (a[states[v],states[u]]) * (b[states[v],mod_sentence[i]])
                except KeyError:
                    alpha[i][u] += 0
        #Beta values
    for i in reversed(range(1,n-1)):
        for u in y:
            for v in y:
                try:
                    beta[i][u] += beta[i+1][v] * (a[states[u],states[v]]) * (b[states[u],mod_sentence[i]])
                except KeyError:
                    beta[i][u] += 0

    #Find the max marginal
    for i in range(n):
        max_p = 0
        for u in y:
            p = alpha[i][u] * beta[i][u]
            if p >= max_p:
                max_p = p
                output_states[i] = states[u]

    return output_states




def maximum_marginal_analysis(language):
    """
    Performs maximum marginal algorithm on our test data
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
    with codecs.open(os.path.join(save_path,'dev.p4.out'), 'w', 'utf-8') as file:
        for sentence in test_data:
            mod_sentence = []
            for word in sentence:
                # To check if word in test data appears in training data
                if word not in x_count or x_count[word] < 3:
                    mod_word = '#UNK#'
                else:
                    mod_word = word

                mod_sentence.append(mod_word)

            output_states = maximum_marginal_sentence(mod_sentence, a, b)

            for i in range(len(sentence)):
                output = sentence[i] + ' ' + output_states[i] + '\n'

                file.write(output)
            file.write('\n')

    print('Done!')
    file.close()

if __name__ == '__main__':
    maximum_marginal_analysis('EN')
    maximum_marginal_analysis('FR')
