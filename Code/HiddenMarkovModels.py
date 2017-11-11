# Machine Learning Design Project
# Team members:
# Jonathan Bei Qi Yang
# Ruth Wong Nam Ying

import os
import sys


class TweetSet:
    """
    Main overall class for a set of tweets
    """

    def __init__(self):
        self.tweets = []
        self.emission = {}
        self.pairs = []
        self.k = 3

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
        Get count of y's that emits the word x in a all tweets
        :return: count of all (y -> x)'s
        """
        val = 0
        for p in self.pairs:
            if ((p[0] == x) and (p[1] == y)):
                val += 1
        return val

    def add_emission_params(self, x, y):
        """
        Get the emission parameter of word label pair and store them
        :return: none
        """
        self.emission[(x, y)] = float(self.count_y_to_x(x, y)) / float(self.count_total_y(y))

    def modify_data(self, k):
        """
        Replace words that appear less than k times in training set with #UNK#
        :return: none
        """
        for tweet in self.tweets:
            for i in range(tweet.getSize()):
                if (self.count_total_x(tweet.get_x()[i]) < 3):
                    self.pairs.append(("#UNK#", tweet.get_y()[i]))
                else:
                    self.pairs.append((tweet.get_x()[i], tweet.get_y()[i]))


    def all_emission_params(self):
        """
        Iterate through all word/label pairs to populate emission parameters list
        :return: none
        """
        self.modify_data(self.k)
        for p in self.pairs:
            self.add_emission_params(p[0], p[1])

    def get_all(self):
        """
        :return: all tweets
        """
        return self.tweets

    def get_emission_params(self):
        """
        :return: all emission parameters
        """
        return self.emission


class HiddenMarkovModel:
    """
    Class container for holding an instance of a tweet
    """

    def __init__(self):
        self.x = []
        self.y = []

    def add_x(self, x):
        """
        Adds a new word to the current HMM model
        :param x: words
        """
        self.x.append(x)

    def add_y(self, y):
        """
        Adds a new label/tag to the current HMM model
        :param y: label/tag
        """
        self.y.append(y)

    def count_x(self,x):
        """
        Get count of x's in a sentence/tweet
        :return: count of x's
        """
        count = 0
        for i in self.x:
            if x == i:
                count += 1

        return count

    def count_y(self,y):
        """
        Get count of y's in a sentence/tweet
        :return: count of y's
        """
        count = 0
        for i in self.y:
            if y == i:
                count += 1

        return count

    def getSize(self):
        """
        Get size of sentence/tweet
        :return: size of tweet
        """
        return len(self.x)

    def get_x(self):
        """
        Get all words in tweet
        :return: list of x
        """
        return self.x

    def get_y(self):
        """
        Get all label in tweet
        :return: list of y
        """
        return self.y


def read_training_set():
    """
    Reads in training data as provided
    """
    main_path = os.path.dirname(__file__)
    training_path = os.path.join(main_path,'../Datasets/Demo/train')
    training = open(training_path, encoding='utf-8').read().split('\n\n')
    del training[-1]    # because the last tweet ends with 3 '\n'
    print('Read training file with ' + str(len(training)) + ' tweets')

    new_set = TweetSet()

    count = 0
    for i in training:
        # print("i: " + i )
        count = count + 1
        print(count)
        new_tweet = HiddenMarkovModel()

        for j in i.split('\n'):
            # print(j.split(" "))
            if len(j) > 0:
                new_tweet.add_x(j.split(" ")[0])
                new_tweet.add_y(j.split(" ")[1])


        if new_tweet.getSize() > 0:
            new_set.add_tweet(new_tweet)

    print("done!")
    return new_set


trainset = read_training_set()
trainset.all_emission_params()
print (trainset.get_emission_params())
