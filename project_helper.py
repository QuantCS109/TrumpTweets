import pandas as pd
from collections import Counter
from pytz import timezone
import re


class TweetData:

    def __init__(self, file='trump_archive_db.csv'):
        self.file = file
        self.raw_data = []
        self.error_tweets = {}

        self.read_data()
        self.raw_tweets = self.parse()
        self.clean_tweets = self.clean()
        self.text = self.create_text()
        self.words = self.tokenize_text()
        self.vocab_to_int, self.int_to_vocab = self.create_lookup_tables()
        self.int_words = self.create_int_words()

    def read_data(self):
        with open(self.file, mode='r') as f:
            for row in f:
                self.raw_data.append(row)

    def parse(self):
        timestamps = []
        tweets = []
        raw_tweets = pd.DataFrame(columns=['tweets'])
        for i, tweet in enumerate(self.raw_data):
            try:
                timestamps.append(timezone('US/Central').localize(pd.to_datetime((tweet[-21:-2]))))
                tweets.append(tweet[:-22])
            except:
                self.error_tweets[i] = tweet
        raw_tweets['tweets'] = tweets
        raw_tweets.index = timestamps
        raw_tweets.index.name = 'timestamp'

        return raw_tweets

    def clean_step_1(self, tweet):

        # Remove whitespace before and after tweet
        tweet = tweet.strip(' ')
        tweet = tweet.lstrip('\"')
        tweet = tweet + '\n\n'
        return tweet

    def clean_step_2(self, tweet):
        # Remove http links
        tweet = re.sub(r"http\S+", '', tweet)
        # Remove hash tags
        tweet = re.sub(r"#\S+", '', tweet)
        # Remove twitter handles
        tweet = re.sub(r"@\S+", '', tweet)
        # Turn everything to lower case
        tweet = tweet.lower()
        # Remove symbols other than letters in the alphabet and numbers
        tweet = re.sub(r"\'", '', tweet)
        tweet = re.sub(r'[^a-zA-Z0-9]', ' ', tweet)
        # Remove whitespace before and after tweet, add one white space
        tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)
        tweet = ' '.join(tweet.split())
        tweet = tweet + ' '
        return tweet

    def clean(self):
        clean_tweets = pd.DataFrame(columns=['tweets'])
        clean_tweets['tweets'] = self.raw_tweets['tweets'].apply(self.clean_step_1)
        clean_tweets['tweets'] = clean_tweets['tweets'].apply(self.clean_step_2)
        clean_tweets.index = self.raw_tweets.index
        return clean_tweets

    def create_text(self):
        text = ''.join(self.clean_tweets.tweets)
        return text

    def tokenize_text(self):
        words = self.text.split()
        # Remove all words with  5 or fewer occurrences
        word_counts = Counter(words)
        return [word for word in words if word_counts[word] > 5]

    def create_lookup_tables(self):

        word_counts = Counter(self.words)
        # words sorted in descending frequency
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
        return vocab_to_int, int_to_vocab

    def create_int_words(self):
        return [self.vocab_to_int[word] for word in self.words]


class APIData(TweetData):
    def __init__(self, file='trumptwits.csv'):
        super().__init__(file=file)

    def read_data(self):
        self.raw_data = pd.read_csv(self.file)

    def parse(self):
        raw_tweets = pd.DataFrame(columns=['tweets'])
        raw_tweets['tweets'] = self.raw_data['text']
        raw_tweets.index = pd.to_datetime(self.raw_data['time'])
        raw_tweets.index.name = 'timestamp'
        raw_tweets = raw_tweets.sort_index()
        raw_tweets = raw_tweets.tz_convert('US/Central')
        return raw_tweets

    def clean_step_1(self, tweet):
        tweet = tweet.lstrip('b\'')
        tweet.rstrip('\n\n\n\n')
        tweet.rstrip('\'')
        return tweet


class MarketData:
    def __init__(self,file='ES.csv'):
        self.file = file
        self.data = self.read_data()

    def read_data(self):
        fin_data = pd.read_csv(self.file)
        fin_data.index = pd.to_datetime(fin_data['Date'] + ' ' + fin_data['Time']).dt.tz_localize('US/Central')
        fin_data.index.name = 'timestamp'
        fin_data.drop(columns=['Date', 'Time'])
        return fin_data



