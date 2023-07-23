import pandas as pd
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Set your path here
path = 'U:\\DataBase\\Chatgpt convo\\'  # You can change this path accordingly

# Load the stop words
stop_words = set(stopwords.words('english'))

# Load the data
df = pd.read_json(os.path.join(path, 'conversations.json'))

# Convert the 'create_time' column to datetime
df['create_time'] = pd.to_datetime(df['create_time'])

# Set the 'create_time' column as the index
df.set_index('create_time', inplace=True)

# Sort the data by the index
df_sorted = df.sort_index()

# Fill the null values in 'title' column with an empty string
df_sorted['title'].fillna('', inplace=True)

# Create a list of words in the 'title' column, excluding stop words
words = [word for word in ' '.join(df_sorted['title']).split() if word not in stop_words]

# Create a counter of words
word_counter = Counter(words)

# Get the most and least common words
most_common_words = word_counter.most_common(10)
least_common_words = word_counter.most_common()[:-11:-1]

# Count the number of unique words in the 'title' column
num_unique_words = len(set(words))

# Extract hour, day and month from create_time and update_time
df_sorted['create_hour'] = df_sorted.index.hour
df_sorted['create_day'] = df_sorted.index.dayofweek
df_sorted['create_month'] = df_sorted.index.month

# Calculate conversation length
df_sorted['conversation_length'] = (df_sorted['update_time'] - df_sorted.index).dt.total_seconds()


# Count the total number of conversations
num_conversations = df.shape[0]

# Count the number of nodes per conversation
df_sorted['num_nodes'] = df_sorted['mapping'].apply(len)
avg_num_nodes = df_sorted['num_nodes'].mean()

# Calculate the length of the longest and shortest conversation
max_convo_length = df_sorted['num_nodes'].max()
min_convo_length = df_sorted['num_nodes'].min()

# Calculate the average length of conversations
avg_convo_length = df_sorted['num_nodes'].mean()

# Get the most common starting and ending words in the 'title' column
starting_words = [title.split()[0] for title in df_sorted['title'] if title and title.split()[0] not in stop_words]
ending_words = [title.split()[-1] for title in df_sorted['title'] if title and title.split()[-1] not in stop_words]
most_common_starting_word = Counter(starting_words).most_common(1)[0][0]
most_common_ending_word = Counter(ending_words).most_common(1)[0][0]

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df_sorted['sentiment'] = df_sorted['title'].apply(lambda x: sia.polarity_scores(x))
df_sorted['sentiment_score'] = df_sorted['sentiment'].apply(lambda x: x['compound'])


# Create a DataFrame for the summary statistics
summary_stats = pd.DataFrame({
    
    'Number of Conversations': [num_conversations],
    'Average Number of Nodes per Conversation': [avg_num_nodes],
    'Most Common Words': [dict(most_common_words)],
    'Least Common Words': [dict(least_common_words)],
    'Length of Longest Conversation': [max_convo_length],
    'Length of Shortest Conversation': [min_convo_length],
    'Average Length of Conversations': [avg_convo_length],
    'Most Common Starting Word': [most_common_starting_word],
    'Most Common Ending Word': [most_common_ending_word],
    'Number of Unique Words': [num_unique_words]
})

# Save the summary statistics to a CSV file
summary_stats.to_csv(os.path.join(path, 'summary_stats.csv'), index=False)

# Just before creating plots that require the 'create_time' field, make sure it's still the index:
assert df_sorted.index.name == 'create_time', "The index should be 'create_time'"

# Sentiment Analysis Plot
df_sorted['sentiment_score'] = df_sorted['sentiment'].apply(lambda x: x['compound'])
plt.figure(figsize=(10, 6))
sns.histplot(df_sorted['sentiment_score'], bins=20, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig(os.path.join(path, 'Distribution of Sentiment Scores.png'))
plt.clf()

# Topic Modeling
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Number of topics you want to find
n_topics = 5

# Use CountVectorizer to get the word counts within each conversation
vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(df_sorted['title'])

# Use LDA to find the topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
lda.fit(word_counts)

# Get the words associated with each topic
words = vectorizer.get_feature_names_out()
topics = dict()
for idx, topic in enumerate(lda.components_):
    topics[idx] = [words[i] for i in topic.argsort()[:-10 - 1:-1]]

# Print the topics
for idx, topic in topics.items():
    print(f'Topic {idx}: {" ".join(topic)}')
    
# Assign each conversation to its dominant topic
topic_assignments = lda.transform(word_counts).argmax(axis=1)
df_sorted['dominant_topic'] = topic_assignments

# Create a DataFrame with the resampled dominant topics
dominant_topics_resampled = df_sorted['dominant_topic'].resample('M').apply(lambda x: x.value_counts().index[0])


# Create a DataFrame with the count of dominant topics for each month
dominant_topics_pivot = df_sorted.pivot_table(index=df_sorted.index.month, columns='dominant_topic', aggfunc='size', fill_value=0)

# Plot the DataFrame
dominant_topics_pivot.plot()
plt.title('Dominant Topics Over Time')
plt.xlabel('Time')
plt.ylabel('Dominant Topic')
plt.legend(title='Dominant Topic', labels=[f'Topic {i}' for i in range(n_topics)])
plt.savefig(os.path.join(path, 'Dominant Topics Over Time.png'))
plt.clf()

df_sorted['sentiment_score'].plot(kind='line')
plt.title('Sentiment Over Time')
plt.xlabel('Time')
plt.savefig(os.path.join(path, 'Sentiment Over Time.png'))
plt.clf()

df_sorted.groupby(df_sorted.index.dayofweek)['sentiment_score'].mean().plot(kind='bar')
plt.title('Average Sentiment by Day of Week')
plt.xticks(ticks=range(7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45)
plt.xlabel('Day of Week')
plt.savefig(os.path.join(path, 'Average Sentiment by Day of Week.png'))
plt.clf()



# Plotting Hourly Distribution of Conversation Creation
plt.figure(figsize=(10, 5))
sns.countplot(data=df_sorted, x='create_hour', color='skyblue')
plt.title('Hourly Distribution of Conversation Creation')
plt.xlabel('Hour of Day')
plt.savefig(os.path.join(path, 'Hourly Distribution of Conversation Creation.png'))
plt.clf()

# Create word cloud of the words in the 'title' column
wordcloud = WordCloud(background_color='white').generate(" ".join(words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.savefig(os.path.join(path, 'wordcloud.png'))
plt.clf()

most_common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])
most_common_words_df.plot(kind='bar', x='Word', y='Frequency', legend=False, title='Most common words')
plt.xlabel('')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(path, 'most_common_words_df.png'))
plt.clf()

df_sorted['create_hour'] = df_sorted.index.hour
df_sorted['create_hour'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.title('Conversations by hour')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversations by hour.png'))
plt.clf()

df_sorted['num_nodes'].plot(kind='hist', bins=20, rwidth=0.9, color='#607c8e')
plt.title('Conversation lengths')
plt.xlabel('Number of nodes')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversation lengths.png'))
plt.clf()

df_sorted.index.value_counts().resample('M').sum().plot(kind='line')
plt.title('Conversations over time')
plt.xlabel('Time')
plt.ylabel('Number of conversations')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversations over time.png'))
plt.clf()

df_sorted['num_nodes'].plot(kind='box')
plt.title('Number of nodes per conversation')
plt.ylabel('Number of nodes')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Number of nodes per conversation.png'))
plt.clf()

df_sorted.plot(kind='scatter', x='num_nodes', y='conversation_length')
plt.title('Conversation length vs. number of nodes')
plt.xlabel('Number of nodes')
plt.ylabel('Conversation length')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversation length vs. number of nodes.png'))
plt.clf()

correlations = df_sorted[['num_nodes', 'conversation_length', 'create_hour']].corr()
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlations')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Heatmap of correlations'))
plt.clf()

df_sorted.index.value_counts().resample('M').sum().plot(kind='area')
plt.title('Conversations over time')
plt.xlabel('Time')
plt.ylabel('Number of conversations')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversations over time.png'))
plt.clf()

sns.violinplot(y=df_sorted['num_nodes'])
plt.title('Conversation lengths')
plt.ylabel('Number of nodes')
plt.tight_layout()
plt.savefig(os.path.join(path, 'Conversation lengths.png'))
plt.clf()

# Sentiment Analysis Plot
df_sorted['sentiment_score'] = df_sorted['sentiment'].apply(lambda x: x['compound'])
plt.figure(figsize=(10, 6))
sns.histplot(df_sorted['sentiment_score'], bins=20, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.savefig(os.path.join(path, 'sentiment_score.png'))
plt.clf()

print("Detailed analysis completed.")
