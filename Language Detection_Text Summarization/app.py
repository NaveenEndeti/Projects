import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from langdetect import detect
from deep_translator import GoogleTranslator
import re
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import networkx as nx

from flask import Flask, request, render_template

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Read data
    dataset = pd.read_csv("D:\KDM\Project\lang_detect\Flask\Lang_detect.csv")
    dataset.head()

    # feature and label extraction
    dataset = dataset.loc[dataset['language'].notna()]
    dataset = dataset.loc[dataset['Text'].notna()]
    dataset["language"].unique()
    dataset = dataset.dropna(axis=1)
    dataset = dataset.values

    # Label encoding
    # Change the text to have the same casing
    dataset[:,0] = [entry.lower() for entry in dataset[:,0]]

    # cleaning the data
    # Tokenize the data
    dataset[:,0] = [word_tokenize(entry) for entry in dataset[:,0]]
    tags = defaultdict(lambda : wn.NOUN)
    tags['J'] = wn.ADJ
    tags['V'] = wn.VERB
    tags['R'] = wn.ADV

    for index, entry in enumerate(dataset[:, 0]):
        words = []
        word_lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
    # dataset[index, 1].lower() in stopwords.fileids() and word not in stopwords.words(dataset[index, 1].lower()) for stop words
            if word.isalpha():   
                final_word = word_lemmatized.lemmatize(word, tags[tag[0]])
                words.append(final_word)
        dataset[index, 0] = str(words)

     # split the dataset
    x_train, x_test, y_train, y_test = train_test_split(dataset[:,0], dataset[:,1], test_size = 0.3, random_state=45)

    vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')
    vectorizer.fit(dataset[:,0])

    train_x_tfidf = vectorizer.transform(x_train)
    test_x_tfidf = vectorizer.transform(x_test)

    sup = svm.SVC()
    sup.fit(train_x_tfidf, y_train)

    pred = sup.predict(test_x_tfidf)
    svm_acc = (accuracy_score(pred, y_test))*100
    print("SVM accuracy: ", svm_acc, '%')

    # Preprocess input from user
    def preprocess(data):
        data = [entry.lower() for entry in data]

        for index, entry in enumerate(data):
            words = []
            word_lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word.isalpha():
                    final_word = word_lemmatized.lemmatize(word, tags[tag[0]])
                    words.append(final_word)
        data[index] = str(words)

        return data
    
    # Predict the language of the data input
    def langIdentify(data):
        predicted = sup.predict(data)
        return predicted
    
    def most_frequent(List):
        dict = {}
        count, word = 0, ''
        for item in reversed(List):
            dict[item] = dict.get(item, 0) + 1
            if dict[item] >= count :
                count, word = dict[item], item
        return(word)
    

    def predict(text):
        words = text.split(" ");
        #x = preprocess(words)
        x = vectorizer.transform(words)
        y = sup.predict(x)
  
  
        final = most_frequent(y)
        print(final)
  
        return final

    if request.method == 'POST':
        txt = request.form['text']
        predict(txt)


    def translate(text):
        if (detect(text) != 'en') or (predict(text) != "English"):
            translated = GoogleTranslator(target='en').translate(text)
            new_text = translated
        else:
            new_text = text
    
        return new_text

    stopWords = set(stopwords.words('english'))

    def sent_similarity(sent1, sent2):
        sent1 = [word.lower() for word in sent1]
        sent2 = [word.lower() for word in sent2]

        total_words = list(set(sent1 + sent2))

        sent1_dist = [0] * len(total_words)
        sent2_dist = [0] * len(total_words)

        for word in sent1:
            if word in stopWords:
                continue
            sent1_dist[total_words.index(word)] += 1 

        for word in sent2:
            if word in stopWords:
             continue
        sent2_dist[total_words.index(word)] += 1 

        return 1 - cosine_distance(sent1_dist, sent2_dist)

    def build_matrix(sentences):
        tfidf = TfidfVectorizer()
        vect_matrix = tfidf.fit_transform(sentences)

        tokens = tfidf.get_feature_names_out()
        matrix = vect_matrix.toarray()

        sent_names = [f'sent_{i+1}' for i, _ in enumerate(matrix)]
        df = pd.DataFrame(data=matrix, index=sent_names, columns = tokens)

        similarity = cosine_similarity(vect_matrix)

        return similarity



    def summarize(matrix, sentences):
        summary = []

        graph = nx.from_numpy_array(matrix)
        scores = nx.pagerank(graph)

        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        for i in range(int(len(sentences)/2)):
            summary.append(ranked[i][1])
        

        return summary
    

    def prep_paragraph(text):
        sentences = sent_tokenize(text)
  
        matrix = build_matrix(sentences)

        summary = summarize(matrix, sentences)

        return summary

    
    text_summerization=[]
    test2 = translate(txt)
    for i in prep_paragraph(test2):
        text_summerization.append(i)
    
    print(text_summerization)
    
    return render_template('index.html',Accuracy='Model Accuracy {}'.format(svm_acc), prediction='Language is in {}'.format(predict(txt)),prediction1='Text Summerization: {}'.format(text_summerization))
           
       
if __name__ == "__main__":
    app.run(debug=True)