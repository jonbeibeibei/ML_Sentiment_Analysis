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
        self.emission = []

    def add_tweet(self, tweet):
        """
        Appends new tweet to overall set
        :param tweet:
        """
        self.tweets.append(tweet)

    def count_total_y(self, y):
        """
        Get count of y's in a all tweets
        :return: count of all y's
        """
        val = 0
        for tweet in self.tweets:
            val += tweet.count_y(y)
        return val

    def add_emission_params(self, x, y):
        """
        Get the emission parameters and store them
        :return: none
        """


    def return_all(self):
        return self.tweets


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

    def return_x(self):
        return self.x

    def return_y(self):
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
print (trainset.count_total_y("O"))
