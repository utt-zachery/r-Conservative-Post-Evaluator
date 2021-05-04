import nltk
import html
import datetime
import string
import re
import numpy
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import FreqDist
import pickle
from tensorflow import keras

class Predictor:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.wordnet.WordNetLemmatizer()
        self.tweet_tokenizer = nltk.TweetTokenizer()
        self.sia = SentimentIntensityAnalyzer()
        self.sampleCount = 1000

        markovHandler = open('model/bayesTable.data', 'rb')
        self.markovTable = pickle.load(markovHandler)
        markovHandler.close()

        vocabHandler = open('model/vocab.data', 'rb')
        self.topWords = pickle.load(vocabHandler)
        vocabHandler.close()

        sentHandler = open('model/avsent.data', 'rb')
        self.avSentiment = pickle.load(sentHandler)
        sentHandler.close()

        self.model = keras.models.load_model('model/best')

    def postTextScraper(self, post):

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

    def generatePostWordFeatures(self, post):
        words = self.postTextScraper(post)
        sentiments = self.postVADERScraper(post)

        postContentFrequency = FreqDist(words)
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
        fullStr = post["title"] + " " + post["selftext"]
        features = numpy.zeros(7 + 24 + 1 + 5 + 1, dtype=float)
        self.tagDayOfWeek(self.ep_to_day(int(post["created_utc"])), features)
        features[7 + self.getHour(int(post["created_utc"]))] = 1
        features[7 + 24] = self.sia.polarity_scores(post["title"] + " " + post["selftext"])["compound"]
        features[7 + 24 + 1] = len(words)
        denom = max(len(words),1)
        features[7 + 24 + 2] = numpy.sum([len(word) for word in words]) / denom
        features[7 + 24 + 3] = len(re.findall(r'[A-Z]', fullStr)) / denom
        features[7 + 24 + 4] = len(re.findall(r'[\.?!]', fullStr)) / denom
        features[7 + 24 + 4 + 1] = self.generateMarkovPrediction(words)
        return features


    def predictPost(self, post):
        postWordFeatures = self.generatePostWordFeatures(post)
        postFeatures = self.generatePostFeatures(post)
        currentPrediction = self.model([postWordFeatures.reshape(1,-1), postFeatures.reshape(1,-1)])

        sampleSet = []
        copySet = []
        activeWords = []
        truePost = scorePost(currentPrediction)

        for index in range(int(len(postWordFeatures)/2), len(postWordFeatures)):
            if postWordFeatures[index] != 0:
                edited = numpy.copy(postWordFeatures)
                edited[index] = 0
                edited[index-self.sampleCount] = 0
                activeWords.append(self.topWords[index-self.sampleCount][0])
                sampleSet.append(edited)
                copySet.append(numpy.copy(postFeatures))

        wordResults = dict()
        if (len(sampleSet) > 0):
            predictions = self.model.predict([numpy.array(sampleSet), numpy.array(copySet)])
            for i in range(0, len(predictions)):
                wordResults[activeWords[i]] = truePost-scorePost(predictions[i])

        highestScore = -1
        bestHour = -1
        bestDay = -1
        for i in range(0, 7+24):
            postFeatures[i] = 0

        constants = []
        trials = []

        sols = []
        for day in range(0, 7):
            for hour in range(0, 24):
                postFeatures[day] = 1
                postFeatures[7+hour] = 1

                constants.append(numpy.copy(postWordFeatures))
                trials.append(numpy.copy(postFeatures))
                postFeatures[day] = 0
                postFeatures[7 + hour] = 0
                sols.append((day, hour))

        predictions = self.model.predict([numpy.array(constants), numpy.array(trials)])
        for i,trial in enumerate(predictions):
            if scorePost(trial) > highestScore:
                        highestScore = scorePost(trial)
                        bestHour = sols[i][1]
                        bestDay = sols[i][0]
        return currentPrediction, bestDay, bestHour, wordResults

def scorePost(np):
    return numpy.max(np) + numpy.argmax(np)