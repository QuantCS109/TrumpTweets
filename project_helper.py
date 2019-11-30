import pandas as pd
import numpy as np
from collections import Counter
from pytz import timezone
import re
from datetime import timedelta
import pickle
import re
from sklearn.ensemble import RandomForestClassifier


class TweetData:

    def __init__(self, file='data/trump_archive_db.csv'):
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

        self.daily_tweets = None
        self.get_daily_tweets()

    def read_data(self):
        with open(self.file, mode='r',errors='ignore') as f:
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

    def get_daily_tweets(self):
        self.clean_tweets['timestamp'] = self.clean_tweets.index
        after_4_tweets = self.clean_tweets.timestamp.dt.hour >= 15
        self.clean_tweets['after4_date'] = self.clean_tweets.timestamp.dt.date
        self.clean_tweets.loc[after_4_tweets, 'after4_date'] = self.clean_tweets.timestamp[after_4_tweets].dt.date\
                                                               + timedelta(days=1)
        self.daily_tweets = self.clean_tweets.groupby('after4_date')['tweets'].apply(lambda x: ' '.join(x))
        self.daily_tweets = self.daily_tweets.to_frame('tweets')
        self.daily_tweets.index.name = 'date'


class APIData(TweetData):

    def __init__(self, file='data/trumptwits.csv'):
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


class IntradayData:

    def __init__(self,file='data/ES_intraday.csv'):
        self.file = file
        self.raw_data = self.read_data()

    def read_data(self):
        fin_data = pd.read_csv(self.file)
        fin_data.index = pd.to_datetime(fin_data['Date'] + ' ' + fin_data['Time']).dt.tz_localize('US/Central')
        fin_data.index.name = 'timestamp'
        fin_data = fin_data.drop(columns=['Date', 'Time'])
        return fin_data

    def get_data(self):
        return self.raw_data[['Open', 'Close']]


class FuturesCloseData:
    def __init__(self, path='data/futures_close.csv'):
        self.instrument_list = ['ES', 'NQ', 'CD', 'EC', 'JY', 'MP', 'TY', 'US', 'C', 'S', 'W', 'CL', 'GC']
        self.df = self.load(path)

    def load(self, path):
        df = pd.read_csv(path)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    def features(self, inst):
        return self.momentum(inst)

    def price(self, inst):
        return self.df[inst]

    def returns(self, inst, start=1, end=2):
        returns = (self.df[inst].shift(-end) - self.df[inst].shift(-start)) / self.df[inst].shift(-start)
        return returns

    def momentum(self, inst, lag=60):
        momo = pd.DataFrame((self.df[inst] - self.df[inst].shift(lag)) / self.df[inst])
        momo = momo.dropna()
        momo.columns += '_{}D'.format(lag)
        return momo

    def log_returns(self):
        return np.log(self.df.shift(-1)) - np.log(self.df)


class VolFeatures:
    def __init__(self, path='features/vol_features.pkl'):
        self.instrument_list = ['ES', 'NQ', 'CD', 'EC', 'JY', 'MP', 'TY', 'US', 'C', 'S', 'W', 'CL', 'GC']
        self.df = self.load(path)
        self.col_dict = {inst: [key for key in self.df.columns if re.match(r"{}_+".format(inst), key)]
                         for inst in self.instrument_list}

    def features(self, inst):
        return self.df[self.col_dict[inst]]

    def load(self, path):
        pickle_in = open(path, "rb")
        vol_pd = pickle.load(pickle_in)
        pickle_in.close()
        return vol_pd.fillna(vol_pd.mean())


class TweetReturnsFeatures(VolFeatures):
    def __init__(self, path='features/tweet_returns_features.csv'):
        super().__init__(path)

    def load(self, path):
        tweet_returns = pd.read_csv(path)
        tweet_returns.set_index('date', inplace=True)
        tweet_returns.index = pd.to_datetime(tweet_returns.index)
        return tweet_returns



class TradeModel:

    def __init__(self, model=RandomForestClassifier, *args, **kwargs):
        self.model = model(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)

    def position(self, X, cutoff=0.55):
        # converting predictions from {0,1} to {-1,1}, short/long
        position = 2 * self.model.predict(X) - 1
        position[self.model.predict_proba(X).max(axis=1) <= cutoff] = 0
        return position

    def _strategy_returns(self, x, y):
        strat_rets = x[:-2] * y[:-2]
        strat_rets_cum = (1 + strat_rets).cumprod()
        return strat_rets, strat_rets_cum

    def strategy_returns(self, X, returns, cutoff=0.55):
        return self._strategy_returns(returns, self.position(X, cutoff))

    def sharpe(self, X, returns, cutoff=0.55):
        rets = self.strategy_returns(X, returns, cutoff)[0]
        return np.mean(rets) / np.std(rets)





