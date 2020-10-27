import gensim
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sks
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
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
#sns.countplot(x='label' , data = data )
#plt.show()

print(data['tweet'].str.len())
lst=data['tweet'].str.len()
print("Hey" , lst)
#sns.histplot( lst  )
#plt.show()

#adding a new coloumn to the data
data['len'] = data['tweet'].str.len()
print(data.head(10))

#applying the data transformations
df=data
print("Before Groupby\n" , data)
data.groupby('label')
print(data.describe())

#Averaging and grouping with respect to the label
df.groupby('len').mean()['label']#.plot.hist(color = 'black', figsize = (6, 4) )
#plt.title('variation of length')
#plt.xlabel('Length')
#plt.show()

#converting the word into to count tokens
cv = CountVectorizer(stop_words='english')
words = cv.fit_transform(data.tweet)
#finding the most frequent word in the dataset
sum_words = words.sum(axis=0)
print(sum_words)
words_freq = [(word ,sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1],reverse=True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
print(frequency.head(30))
#plotting the most frequent word
frequency.head()#.plot(x='word', y='freq', kind ='bar', figsize=(6, 4),color ='blue')
#plt.title("Most frequently occuring words - TOP 30")
#plt.show()

#plotting the most frequently used worf through wordcloud
wordcloud= WordCloud (background_color='white', width =1000, height = 1000 ).generate_from_frequencies(dict(words_freq))
#plt.figure(figsize=(10, 8))
#plt.imshow(wordcloud)
#plt.title("Vocabulary from reveiws")
#plt.show()

#plotting the common words
normal_words = ' '.join([text for text in data['tweet'][data['label']== 0]])
wordcloud = WordCloud (background_color = 'black', width=1000, height = 1000,random_state=0,max_font_size=110).generate(normal_words)
#plt.figure(figsize=(10, 7))
#plt.imshow(wordcloud,interpolation="bilinear")
#plt.axis('off')
#plt.title('Neutral words ')
#plt.show()

#plotting the negative words
negative_words  = ' '.join([text for text in data['tweet'][data['label']== 1 ]])
wordcloud = WordCloud (background_color='pink',width = 1000,height=1000,random_state=0,max_font_size=110).generate(negative_words)
#plt.figure(figsize = (10, 7))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.title("Negative words")
#plt.show()

#collecting the hashtags
def hashtag_extract(x):
    hashtags = []

    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
#separting non racism tweets
HT_regular =hashtag_extract(data['tweet'][data['label'] == 0])
#print(HT_regular)
#separting racism/sexist tweets
HT_negative = hashtag_extract(data['tweet'][data['label'] == 1])
#print(HT_negative)
HT_regular= sum(HT_regular,[])
HT_negative= sum(HT_negative, [])

#segregating the most frequently used hashtag
#regular hashtags
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'count': list(a.values())})
d =d.nlargest(columns='count',n=20)
#plt.figure(figsize=(16, 5))
# ax = sns.barplot(data=d, x = "Hashtag", y = "count", )
# ax.set_ylabel("count")
# plt.show()

#negative hashtags
a = nltk.FreqDist(HT_negative)
d = pd.DataFrame({'Hashtag': list(a.keys()),'count': list(a.values())})
d = d.nlargest(columns='count', n=20)
# plt.figure(figsize=(10, 4))
# ax = sns.barplot(data=d, x='Hashtag',y='count')
# ax.set_ylabel("count ")
#plt.show()
#tokenizing the words in the datset
tokenized_tweet = data['tweet'].apply(lambda x: x.split())
#creating a word to vector model
from gensim.models import  word2vec
model_w2v = gensim.models.Word2Vec(tokenized_tweet, size = 200,window = 5, min_count = 2,sg = 1,hs =2, negative  = 10 , workers =4 , seed  = 34)
model_w2v.train(tokenized_tweet,total_examples = len(data['tweet']),epochs = 20 )
#print(model_w2v)

#testing the dataset using the worf classification
test_dine = model_w2v.wv.most_similar(positive= 'dinner')
print(test_dine)
test_milk = model_w2v.wv.most_similar(positive =  'milk')
print(test_milk)
test_hate = model_w2v.wv.most_similar(negative = "hate")
print(test_hate)































