# Machine Learning Design Project
# Team members:
# Jonathan Bei Qi Yang
# Ruth Wong Nam Ying

import os
import sys
import codecs

class HMM:
    def __init__(self):
        self.states = ['B-positive', 'B-neutral', 'B-negative', 'I-positive', 'I-neutral', 'I-negative','O']
        self.output = []
        self.output_train = []

    def viterbi_algorithm(self):
        """
        Sentiment analysis based on the Viterbi algorithm
        Saves file 'dev.p2.out'
        :return: none
        """
        

    def simple_sentiment_analysis(self):
        """
        Sentiment analysis based on emission parameters only
        Saves file 'dev.p2.out'
        :return: none
        """
        train_path = '../Datasets/Demo/train'
        training_set = read_training_set(train_path)
        training_set.all_emission_params()
        emission_params = training_set.get_emission_params()
        train_words = training_set.get_words()

        #Save train_set with modified data

        for tweet_0 in training_set.get_all():
            modified_train_sentence = []
            for pair_0 in tweet_0.get_tweet():
                modified_train_word = pair_0[2]
                response = pair_0[1]

                modified_train_sentence.append([modified_train_word, response])
            self.output_train.append(modified_train_sentence)
            self.save_training_file(self.output_train)


        test_path = '../Datasets/Demo/dev.in'
        test_set = read_test_set(test_path)
        test_set.modify_test_data(train_words)

        for tweet in test_set.get_all():
            sentence = []
            for pair in tweet.get_tweet():
                actual_word = pair[0]
                modified_word = pair[2]
                probability = 0.0
                prediction = ""
                for state in self.states:
                    try:
                        emission = emission_params[(modified_word, state)]
                        if (emission >= probability): # take the larger probability
                            probability = emission
                            prediction = state
                    except:
                        pass

                sentence.append([actual_word, prediction])
            self.output.append(sentence)
            self.save_file(self.output)

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


    def save_training_file(self, result):
        """
        Saves given input into 'modified.train' file on given path
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
        with codecs.open(os.path.join(save_path, 'modified.train'), 'w', 'utf-8') as file:
            file.write(output)

class TweetSet:
    """
    Main overall class for a set of tweets
    """

    def __init__(self):
        self.tweets = []
        self.emission = {}
        self.transition = {}
        self.words = []
        self.k = 3

    def set_k(k):
        self.k = k

    def get_k(k):
        return self.k

    def add_tweet(self, tweet):
        """
        Appends new tweet to overall set
        :param tweet:
        """
        self.tweets.append(tweet)

    def count_total_x(self, x):
        """
        Get count of x's in a all tweets
        :return: count of all x's
        """
        val = 0
        for tweet in self.tweets:
            val += tweet.count_x(x)
        return val

    def count_total_y(self, y):
        """
        Get count of y's in a all tweets
        :return: count of all y's
        """
        val = 0
        for tweet in self.tweets:
            val += tweet.count_y(y)
        return val

    def count_y_to_x(self, x, y):
        """
        Get count of y's that emits the mod_word x in a all tweets
        :return: count of all (y -> x)'s
        """
        val = 0
        for tweet in self.tweets:
            for pair in tweet.get_tweet():
                if(pair[2] == x) and (pair[1] == y):
                    val += 1
        return val

    def count_y_to_y(self, yi, yj):
        """
        Get count of yi's that transits to yj in a all tweets
        :return: count of all (yi, yj)'s
        """
        val = 0
        for tweet in self.tweets:
            sentence = tweet.get_tweet()
            for i in range(tweet.get_size()):
                if (i == 0):
                    if (yi == 'START') and (yj == sentence[i][1]):
                        # when the first label corresponds to yj
                        val += 1
                elif (i == tweet.get_size() - 1):
                    if (yi == sentence[i][1]) and (yj == 'STOP'):
                        # when the last label corresponds to yi
                        val += 1
                elif (yi == sentence[i-1][1]) and (yj == sentence[i][1]):
                    # when the previous label corresponds to yi and the next label corresponds to yj
                    val += 1
        return val

    def add_emission_params(self, x, y):
        """
        Get the emission parameter of word label pair and store them
        :returns: none
        """
        self.emission[(x, y)] = float(self.count_y_to_x(x, y)) / float(self.count_total_y(y))

    def add_transition_params(self, yi, yj):
        """
        Get the transition parameter of label i and label j and store them
        :returns: none
        """
        if (yi == 'START') or (yj == 'STOP'):
            self.transition[(yi, yj)] = float(self.count_y_to_y(yi, yj)) / float(self.get_size())
        else:
            self.transition[(yi, yj)] = float(self.count_y_to_y(yi, yj)) / float(self.count_total_y(yi))

    def modify_train_data(self, k):
        """
        Replace words that appear less than k times in training set with #UNK#
        Appends word to a list for test data modification
        :returns: none
        """
        for tweet in self.tweets:
            for i,pair in enumerate(tweet.get_tweet()):
                if (self.count_total_x(pair[0]) < k):
                    tweet.set_z("#UNK#",i)
                else:
                    self.words.append(pair[0])

    def modify_test_data(self, train_words):
        """
        Replace words from test set that don't appear in train set with #UNK#
        :return: none
        """
        for tweet in self.tweets:
            for i,pair in enumerate(tweet.get_tweet()):
                if (pair[0] not in train_words):
                    tweet.set_z("#UNK#", i)

    def get_words(self):
        return self.words

    def all_emission_params(self):
        """
        Iterate through all mod_word/label pairs to populate emission parameters list
        :return: none
        """
        self.modify_train_data(self.k)
        for tweet in self.tweets:
            for pair in tweet.get_tweet():
                self.add_emission_params(pair[2], pair[1])

    def all_transition_params(self):
        """
        Iterate through tweets to populate transition parameters list
        :return: none
        """
        for tweet in self.tweets:
            sentence = tweet.get_tweet()
            for i in range(tweet.get_size()):
                if (i == 0):
                    # first label in the sentence
                    self.add_transition_params('START', sentence[i][1])
                elif (i == tweet.get_size() - 1):
                    # last label in the sentence
                    self.add_transition_params(sentence[i][1], 'STOP')
                else:
                    self.add_transition_params(sentence[i-1][1], sentence[i][1])

    def get_all(self):
        """
        :return: all tweets
        """
        return self.tweets

    def get_size(self):
        """
        :return: number of tweets
        """
        return len(self.tweets)

    def get_emission_params(self):
        """
        :return: all emission parameters
        """
        return self.emission

    def get_transition_params(self):
        """
        :return: all transition parameters
        """
        return self.transition


class Tweet:
    """
    Class container for holding an instance of a tweet
    """

    def __init__(self):
        self.tweet = []

    def get_tweet(self):
        """
        :returns: a list of all word/label pairs in the tweet
        """
        return self.tweet

    def get_pair(self, index):
        """
        :returns: word/label pair of given index
        """
        return self.tweet[index]

    def set_pair(self, x, y, z):
        """
        Sets a word/label/mod_word pair within a tweet
        :returns: none
        """
        self.tweet.append([x,y,z])

    def set_x(self, x, i):
        """
        Sets a word for given index
        :returns: none
        """
        self.tweet[i][0] = x

    def set_y(self, y, i):
        """
        Sets a label for given index
        :returns: none
        """
        self.tweet[i][1] = y

    def get_all_x(self):
        """
        :returns: all words from a tweet
        """
        output = []
        for p in self.tweet:
            output.append(p[0])
        return output

    def get_all_y(self):
        """
        :returns: all labels from a tweet
        """
        output = []
        for p in self.tweet:
            output.append(p[1])
        return output

    def get_size(self):
        """
        :returns: size of tweet
        """
        return len(self.tweet)

    def count_x(self,x):
        """
        Get count of x's in a sentence/tweet
        :return: count of x's
        """
        count = 0
        for i in self.tweet:
            if x == i[0]:
                count += 1

        return count

    def count_y(self,y):
        """
        Get count of y's in a sentence/tweet
        :return: count of y's
        """
        count = 0
        for i in self.tweet:
            if y == i[1]:
                count += 1

        return count

    def set_z(self, z, i):
        """
        Modify the words according to appearance
        :returns: none
        """
        self.tweet[i][2] = z

    def get_z(self, i):
        """
        :returns: modified word of given index
        """
        return self.tweet[i][2]

def read_training_set(path):
    """
    Reads in training data as provided
    """
    main_path = os.path.dirname(__file__)
    training_path = os.path.join(main_path, path)
    training = open(training_path, encoding='utf-8').read().split('\n\n')
    del training[-1]    # because the last tweet ends with 3 '\n'
    print('Read training file with ' + str(len(training)) + ' tweets')

    new_set = TweetSet()

    count = 0
    for i in training:
        # print("i: " + i )
        count = count + 1
        new_tweet = Tweet()

        j = i.split('\n')

        for j in i.split('\n'):
            # print(j.split(" "))
            if len(j) > 0:
                new_tweet.set_pair(j.split(" ")[0], j.split(" ")[1], j.split(" ")[0])

        if new_tweet.get_size() > 0:
            new_set.add_tweet(new_tweet)

    print("done!")
    return new_set

def read_test_set(path):
    """
    Reads in test data as provided
    """
    main_path = os.path.dirname(__file__)
    training_path = os.path.join(main_path, path)
    training = open(training_path, encoding='utf-8').read().split('\n\n')
    del training[-1]    # because the last tweet ends with 3 '\n'
    print('Read test file with ' + str(len(training)) + ' tweets')

    new_set = TweetSet()

    count = 0
    for i in training:
        # print("i: " + i )
        count = count + 1
        new_tweet = Tweet()

        for j in i.split('\n'):
            # print(j.split(" "))
            if len(j) > 0:
                new_tweet.set_pair(j, "", j)


        if new_tweet.get_size() > 0:
            new_set.add_tweet(new_tweet)

    print("done!")
    return new_set


# h = HMM()
# output = h.simple_sentiment_analysis()
# print(output)

train_path = '../Datasets/Demo/train'
t = read_training_set(train_path)
t.all_transition_params()
print (t.get_transition_params())
