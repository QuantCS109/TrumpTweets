from sklearn.cluster import KMeans
import pandas as pd
from project_helper import TweetData

tweet_data = TweetData()
topics_df = tweet_data.clean_tweets[tweet_data.clean_tweets.after4_date >= pd.to_datetime('1-1-2017')]

emb = pd.read_csv('tweet_embeddings.csv',index_col=0)
kmeans = KMeans(n_clusters=10, random_state=0).fit(emb)
topics = kmeans.predict(emb)

topics_df['topic'] = kmeans.predict(emb)

topics_df.groupby('topic').agg('count')

topics_analysis = pd.DataFrame()
topics_analysis['tweet_list'] = topics_df.tweets.str.split(' ')
topics_analysis['topic'] = topics_df['topic']
topics_analysis_melt = topics_analysis.explode('tweet_list')
topics_analysis_agg = topics_analysis_melt.assign(topic_count=1).groupby(['tweet_list','topic']).agg('count').reset_index()
all_count = topics_analysis_melt.groupby('tweet_list').agg(all_count=pd.NamedAgg('topic','count'))
topics_analysis_joined = topics_analysis_agg.join(all_count,on='tweet_list')
topics_analysis_joined['prop'] = topics_analysis_joined.topic_count/ topics_analysis_joined.all_count
topics_analysis_joined.to_csv('topics_analysis_agg.csv')