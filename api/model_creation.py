import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nltk.tokenize import RegexpTokenizer  
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.pipeline import make_pipeline
from PIL import Image
import pickle 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split


df= pd.read_csv("./data.csv")

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
tokenizer.tokenize(df.URL[0])

df['text_tokenized'] = df.URL.map(lambda t: tokenizer.tokenize(t))
stemmer = SnowballStemmer("english") 
df['text_stemmed'] = df['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
df['text_sent'] = df['text_stemmed'].map(lambda l: ' '.join(l))

bad_sites = df[df.Label == 'bad']
good_sites = df[df.Label == 'good']

cv = CountVectorizer()
feature = cv.fit_transform(df.text_sent)

feature[:5].toarray() 


trainX, testX, trainY, testY = train_test_split(feature, df.Label)

lr = LogisticRegression()

pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())

trainX, testX, trainY, testY = train_test_split(df.URL, df.Label)

pipeline_ls.fit(trainX,trainY)

pipeline_ls.score(testX,testY)

print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))



pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


