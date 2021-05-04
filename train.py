import json
import pickle
import random
import nltk
import html
import numpy
import datetime
import string
import sys
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import tensorflow
import keras
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input

class ModelTrainer:
    def __init__(self, subreddit):
        self.subReddit = subreddit
        self.allPosts = []
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.wordnet.WordNetLemmatizer()
        self.tweet_tokenizer = nltk.TweetTokenizer()
        self.sia = SentimentIntensityAnalyzer()
        self.sampleCount =1000
        self.avSentiment = 0

    def load(self):
        text_file = open(self.subReddit + ".txt", "r")
        self.allPosts = json.load(text_file)
        text_file.close()

    def train(self):
        
        random.shuffle(self.allPosts)

        prop_vector = [0.9, 0.1]
        self.modelTrainingSet = self.allPosts[0:int(len(self.allPosts) * prop_vector[0])]
        self.testSet = self.allPosts[len(self.modelTrainingSet):]

        print("Model Training Size: {}".format(len(self.modelTrainingSet)))
        print("Test Size: {}".format(len(self.testSet)))

        from nltk import FreqDist
        relevanceTable = FreqDist()

        self.avSentiment = 0
        totalCompound = 0
        for post in self.modelTrainingSet:
            words = self.postTextScraper(post)
            for word in words:
                relevanceTable[word] += 1
            totalCompound += self.sia.polarity_scores(post["title"] + " " + post["selftext"])["compound"]

        self.avSentiment = totalCompound / len(self.modelTrainingSet)

        sampleCount = 1000
        self.topWords = relevanceTable.most_common(sampleCount)

        topDict = set([word[0] for word in self.topWords])
        bayesTable = dict()
        for post in self.modelTrainingSet:
            words = self.postTextScraper(post)
            prevWord = None
            for word in words:
                if word in topDict:
                    if prevWord is not None:
                        if prevWord in bayesTable:
                            if word in bayesTable[prevWord]:
                                bayesTable[prevWord][word].append(post["score"])
                            else:
                                bayesTable[prevWord][word] = [post["score"]]
                        else:
                            bayesTable[prevWord] = dict()
                            bayesTable[prevWord][word] = [post["score"]]
                    prevWord = word

        filehandler2 = open("model/vocab.data", 'wb')
        pickle.dump(self.topWords, filehandler2)
        filehandler2.close()

        filehandler3 = open("model/avsent.data", 'wb')
        pickle.dump(self.avSentiment, filehandler3)
        filehandler3.close()

        avPost = numpy.mean([post["score"] for post in self.modelTrainingSet])
        stDevPost = numpy.std([post["score"] for post in self.modelTrainingSet])

        self.markovTable = dict()
        for word in bayesTable:
            for nextWord in bayesTable[word]:
                if (len(bayesTable[word][nextWord]) > 0):
                    if word in self.markovTable:
                        self.markovTable[word][nextWord] = ((numpy.mean(bayesTable[word][nextWord]) - avPost) / stDevPost)
                    else:
                        self.markovTable[word] = dict()
                        self.markovTable[word][nextWord] = ((numpy.mean(bayesTable[word][nextWord]) - avPost) / stDevPost)

        filehandler = open("model/bayesTable.data", 'wb')
        pickle.dump(self.markovTable, filehandler)
        filehandler.close()

        train_feature1, train_feature2, train_labels = self.convertToFeatureRep(self.modelTrainingSet)
        test_feature1, test_feature2, test_labels = self.convertToFeatureRep(self.testSet)

        testIndex = int(len(test_feature1) / 2)

        val_features1 = test_feature1[0:testIndex]
        test_feature1 = test_feature1[testIndex:]
        val_features2 = test_feature2[0:testIndex]
        test_feature2 = test_feature2[testIndex:]
        val_labels = test_labels[0:testIndex]
        test_labels = test_labels[testIndex:]

        initializer = tensorflow.keras.initializers.HeUniform()

        inputA = Input(shape=(len(train_feature1[0]),))
        fc1 = Dense(len(train_feature1[0]), activation='relu', kernel_initializer=initializer)(inputA)
        fc15 = Dense(int((sampleCount + len(train_feature1[0])) / 2), activation='relu',
                     kernel_initializer=initializer)(fc1)
        fc2 = Dense(sampleCount, activation='relu', kernel_initializer=initializer)(fc15)
        branch1 = Model(inputs=inputA, outputs=fc2)

        inputB = Input(shape=(len(train_feature2[0]),))
        fc1B = Dense(len(train_feature2[0]), activation='relu', kernel_initializer=initializer)(inputB)
        fc2B = Dense(len(train_feature2[0]) / 2, activation='relu', kernel_initializer=initializer)(fc1B)
        branch2 = Model(inputs=inputB, outputs=fc2B)

        combined = keras.layers.concatenate([branch1.output, branch2.output])
        fc1C = Dense(512, activation='relu', kernel_initializer=initializer)(combined)
        fc1C2 = Dense(256, activation='relu', kernel_initializer=initializer)(fc1C)
        fc2C = Dense(128, activation='relu', kernel_initializer=initializer)(fc1C2)

        d3c = Dropout(0.25)(fc2C)
        fc4C = Dense(64, activation='relu', kernel_initializer=initializer)(d3c)
        fc5C = Dense(32, activation='relu', kernel_initializer=initializer)(fc4C)
        fc6C = Dense(16, activation='relu', kernel_initializer=initializer)(fc5C)
        fc7C = Dense(8, activation='relu', kernel_initializer=initializer)(fc6C)
        fc8C = Dense(4, activation='relu', kernel_initializer=initializer)(fc7C)
        fc9C = Dense(4, activation='softmax', kernel_initializer=initializer)(fc8C)
        opt = keras.optimizers.Adam(learning_rate=0.00007)

        model = Model(inputs=[branch1.input, branch2.input], outputs=fc9C)

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

        from keras.callbacks import EarlyStopping
        e1 = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=4, restore_best_weights=True)
        model.fit([train_feature1, train_feature2], train_labels, epochs=50, callbacks=[e1],
                  validation_data=([val_features1, val_features2], val_labels))

        model.save("model/best")
        print(model.evaluate([test_feature1, test_feature2], test_labels))

    def postTextScraper(self,post):
        if (type(post) is str):
            return [self.stemmer.lemmatize(word.lower()) for word in self.tweet_tokenizer.tokenize(post) if
                    word not in self.stop_words and word not in string.punctuation and word != "’" and word != "‘"]

        toReturn = []
        title = str(post["title"])
        sentences = nltk.sent_tokenize(html.unescape(title))

        for sentenceSample in sentences:
            words = [self.stemmer.lemmatize(word.lower()) for word in self.tweet_tokenizer.tokenize(sentenceSample) if
                     word not in self.stop_words and word not in string.punctuation and word != "’" and word != "‘"]
            toReturn.extend(words)

        title = str(post["selftext"])
        sentences = nltk.sent_tokenize(html.unescape(title))

        for sentenceSample in sentences:
            words = [self.stemmer.lemmatize(word.lower()) for word in self.tweet_tokenizer.tokenize(sentenceSample) if
                     word not in self.stop_words and word not in string.punctuation and word != "’" and word != "‘"]
            toReturn.extend(words)

        return toReturn

    def postVADERScraper(self, post):
        toReturn = {}

        title = ""
        if (type(post) is str):
            title = post
        else:
            title = str(post["title"])

        sentences = nltk.sent_tokenize(html.unescape(title))
        for sentenceSample in sentences:
            score = self.sia.polarity_scores(sentenceSample)["compound"]
            words = [self.stemmer.lemmatize(word.lower()) for word in self.tweet_tokenizer.tokenize(sentenceSample) if
                     word not in self.stop_words and word not in string.punctuation and word != "’" and word != "‘"]
            for word in words:
                toReturn[word] = score

        if (type(post) is str):
            return toReturn

        title = str(post["selftext"])
        sentences = nltk.sent_tokenize(html.unescape(title))

        for sentenceSample in sentences:
            score = self.sia.polarity_scores(sentenceSample)["compound"]
            words = [self.stemmer.lemmatize(word.lower()) for word in self.tweet_tokenizer.tokenize(sentenceSample) if
                     word not in self.stop_words and word not in string.punctuation and word != "’" and word != "‘"]
            for word in words:
                if word in toReturn:
                    toReturn[word] = (toReturn[word] + score) / 2
                else:
                    toReturn[word] = score

        return toReturn



    def ep_to_day(self, ep):
        return datetime.date.fromtimestamp(ep / 1000).strftime("%A")

    def getHour(self, ep):
        return int(datetime.date.fromtimestamp(ep / 1000).strftime("%H"))

    def tagDayOfWeek(self, day, array):
        if day == 'Sunday':
            array[0] = 1
        elif day == 'Monday':
            array[1] = 1
        elif day == 'Tuesday':
            array[2] = 1
        elif day == 'Wednesday':
            array[3] = 1
        elif day == 'Thursday':
            array[4] = 1
        elif day == 'Friday':
            array[5] = 1
        elif day == 'Saturday':
            array[6] = 1

    def generatePostWordFeatures(self,post):
        words = self.postTextScraper(post)
        sentiments = self.postVADERScraper(post)

        postContentFrequency = nltk.FreqDist(words)
        features = numpy.zeros(self.sampleCount * 2, dtype=float)
        for index in range(0, self.sampleCount):
            if self.topWords[index][0] in postContentFrequency:
                features[index] = (sentiments[self.topWords[index][0]]) - (self.avSentiment)
                features[index + self.sampleCount] = (postContentFrequency[self.topWords[index][0]]) / len(words)
        return features

    def generateMarkovPrediction(self, words):
        totalPredict = 0
        totalNum = 0
        for index in range(0, len(words) - 1):
            if words[index] in self.markovTable and words[index + 1] in self.markovTable[words[index]]:
                totalPredict += self.markovTable[words[index]][words[index + 1]]
                totalNum += 1

        if totalNum == 0:
            return 0
        return totalPredict / totalNum

    def generatePostFeatures(self, post):
        words = self.postTextScraper(post)
        sentiments = self.postVADERScraper(post)
        fullStr = post["title"] + " " + post["selftext"]
        postContentFrequency = nltk.FreqDist(words)
        features = numpy.zeros(7 + 24 + 1 + 5, dtype=float)
        self.tagDayOfWeek(self.ep_to_day(int(post["created_utc"])), features)
        features[7 + self.getHour(int(post["created_utc"]))] = 1
        features[7 + 24] = self.sia.polarity_scores(post["title"] + " " + post["selftext"])["compound"]
        features[7 + 24 + 1] = len(words)
        features[7 + 24 + 2] = numpy.sum([len(word) for word in words]) / len(words)
        features[7 + 24 + 3] = len(re.findall(r'[A-Z]', fullStr)) / len(words)
        features[7 + 24 + 4] = len(re.findall(r'[\.?!]', fullStr)) / len(words)
        features[7 + 24 + 4 + 1] = self.generateMarkovPrediction(words)
        return features

    def convertToFeatureRep(self, listPost):
        Q1 = numpy.percentile([post["score"] for post in self.allPosts], 25, interpolation='midpoint')
        Q2 = numpy.percentile([post["score"] for post in self.allPosts], 50, interpolation='midpoint')
        Q3 = numpy.percentile([post["score"] for post in self.allPosts], 75, interpolation='midpoint')

        feature1 = []
        feature2 = []
        labels = []

        for post in listPost:
            feature1.append((self.generatePostWordFeatures(post)))
            feature2.append((self.generatePostFeatures(post)))
            if int(post["score"]) < Q1:
                labels.append(numpy.array([1, 0, 0, 0]))
            elif int(post["score"]) < Q2:
                labels.append(numpy.array([0, 1, 0, 0]))
            elif int(post["score"]) < Q3:
                labels.append(numpy.array([0, 0, 1, 0]))
            else:
                labels.append(numpy.array([0, 0, 0, 1]))

        return numpy.array(feature1), numpy.array(feature2), numpy.array(labels)
        
if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Incorrect Syntax. Please run train.py <subreddit name>")
    

    model = ModelTrainer(sys.argv[1])
    model.load()
    model.train()