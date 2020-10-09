import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sks
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
#importing data
data = pd.read_csv('train_tweet.csv')
#checking data
#print(data.head())
print(data.shape)
print(data.info())

#checking data for null values
print(data.isnull().sum())

#checking the positive tweets from the train set

print(data[data['label'] == 1].head(10))

#checking the negative tweets from the train set
print(data[data['label'] == 0].head(10))

#counting the data points

print(data['label'].value_counts())

#performing eda on the data
sns.countplot(x='label' , data = data )
plt.show()

print(data['tweet'].str.len())
lst=data['tweet'].str.len()
print("Hey" , lst)
sns.histplot( lst  )
plt.show()

#adding a new coloumn to the data
data['len'] = data['tweet'].str.len()
print(data.head(10))

#applying the data transformations
df=data
print("Before Groupby\n" , data)
data.groupby('label')
print(data.describe())

#Averaging and grouping with respect to the label
df.groupby('len').mean()['label'].plot.hist(color = 'black', figsize = (6, 4) )
plt.title('variation of length')
plt.xlabel('Length')
plt.show()

#converting the word into to count tokens
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(data.tweet)
sum_words = words.sum(axis=0)
print(sum_words)
#
words_freq = [(word ,sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1],reverse=True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

print(frequency.head(30))
frequency.head().plot(x='word', y='freq', kind ='bar', figsize=(10, 4),color ='blue')
plt.title("Most frequently occuring words - TOP 30")
plt.show()









