import sys
from math import floor

import nltk
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QPlainTextEdit, QVBoxLayout
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QHBoxLayout
import calendar
import time
import numpy
from Predicter import Predictor


class ListeningTextEditor(QtWidgets.QTextEdit):
    def __init__(self, textLabel, predictor, predictionThermo, confidenceThermo,
                 dayLabel, hourLabel):
        QtWidgets.QTextEdit.__init__(self)
        self.textLabel = textLabel
        self.predictor = predictor
        self.predictionThermo = predictionThermo
        self.confidenceThermo = confidenceThermo
        self.dayLabel = dayLabel
        self.hourLabel = hourLabel
        self.tweet_tokenizer = nltk.TweetTokenizer()
        self.stemmer = nltk.wordnet.WordNetLemmatizer()

    def keyPressEvent(self, event):
        if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Return and self.hasFocus():

            self.textLabel.repaint()
            self.textLabel.setText(self.toPlainText())
            post = dict()
            post["title"] = self.toPlainText()
            post["selftext"] = ""
            post["created_utc"] = calendar.timegm(time.gmtime())
            predict, bestDay, bestHour, wordResults = predictor.predictPost(post)

            sentences = nltk.sent_tokenize(self.toPlainText())
            setString = """<p style="line-height:50px; size: 20px;">"""

            for sentenceSample in sentences:
                words = self.tweet_tokenizer.tokenize(sentenceSample)
                for word in words:
                    normalized = self.stemmer.lemmatize(word.lower())
                    if (normalized in wordResults) and abs(wordResults[normalized]) > 0.5:
                        if wordResults[normalized] <0:
                            setString += """<span style="background-color:rgba(229, 64, 94, """ + str(((-1)*wordResults[normalized])/4) + """)">""" + word + """</span> """
                        else:
                            setString += """<span style="background-color:rgba(63, 255, 162, """ + str((wordResults[normalized]) / 4) + """)">""" + word + """</span> """
                    else:
                        setString += """<span>""" + word + """</span> """

            setString += """</p>"""
            self.textLabel.setText(setString)
            np = predict.numpy()
            self.predictionThermo.setThermometer(numpy.argmax(np))
            self.confidenceThermo.setThermometer(int(floor(numpy.max(np) / 0.25)))
            self.dayLabel.setText(["Sundays","Mondays","Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays"][bestDay])
            self.hourLabel.setText(str(bestHour) + ":00 UTC")
        else:
            QtWidgets.QTextEdit.keyPressEvent(self,event)



class Thermometer(QtWidgets.QGridLayout):
    def  __init__(self, labString1, labString2):
        super(Thermometer, self).__init__()
        self.red = QtWidgets.QLabel()
        self.red.setBackgroundRole(True)
        self.red.setStyleSheet("background-color: #e5405e")
        self.red.setFixedWidth(40)
        self.red.setFixedHeight(20)

        self.orange = QtWidgets.QLabel()
        self.orange.setBackgroundRole(True)
        self.orange.setStyleSheet("background-color: darkorange")
        self.orange.setFixedWidth(40)
        self.orange.setFixedHeight(20)

        self.yellow = QtWidgets.QLabel()
        self.yellow.setBackgroundRole(True)
        self.yellow.setStyleSheet("background-color: #ffdb3a")
        self.yellow.setFixedWidth(40)
        self.yellow.setFixedHeight(20)

        self.green = QtWidgets.QLabel()
        self.green.setBackgroundRole(True)
        self.green.setStyleSheet("background-color: #3fffa2")
        self.green.setFixedWidth(40)
        self.green.setFixedHeight(20)
        self.green.setAlignment(Qt.AlignHCenter)

        self.arrow = QtWidgets.QLabel()
        pixmap = QPixmap('resources/arrow.svg')
        self.arrow.setPixmap(pixmap)
        self.arrow.setAlignment(Qt.AlignHCenter)

        lowLab = QLabel()
        lowLab.setText(labString1)
        lowLab.setAlignment(Qt.AlignRight)
        highLab = QLabel()
        highLab.setText(labString2)
        highLab.setAlignment(Qt.AlignLeft)

        self.addWidget(lowLab, 0, 0)
        self.addWidget(self.red, 0, 1)
        self.addWidget(self.orange, 0, 2)
        self.addWidget(self.yellow, 0, 3)
        self.addWidget(self.green, 0, 4)
        self.addWidget(highLab, 0, 5)
        self.addWidget(self.arrow, 1, 3)

    def setThermometer(self, toSet):
        self.orange.setStyleSheet("background-color: gray")
        self.yellow.setStyleSheet("background-color: gray")
        self.green.setStyleSheet("background-color: gray")
        self.removeWidget(self.arrow)

        if (toSet >= 1):
            self.orange.setStyleSheet("background-color: darkorange")
        if (toSet >= 2):
            self.yellow.setStyleSheet("background-color: #ffdb3a")
        if (toSet >= 3):
            self.green.setStyleSheet("background-color: #3fffa2")

        self.addWidget(self.arrow,1,toSet+1)

if __name__ == "__main__":
    predictor = Predictor()
    #predictor = None

    app = QApplication(sys.argv)
    window = QWidget()
    app.setStyle("Fusion")
    window.setWindowTitle("CIS4920 NLP Final Project - Zachery Utt")

    # Dark theme comes from Stack Overflow: https://stackoverflow.com/questions/48256772/dark-theme-for-qt-widgets
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)
    feedback = QLabel()

    upperLayout = QVBoxLayout()

    therm1 = Thermometer("Low Scoring", "High Scoring")
    therm2 = Thermometer("Low Confidence", "High Confidence")
    bestDOW = QLabel()
    bestTIME = QLabel()

    layout = QHBoxLayout()
    textEdit = ListeningTextEditor(feedback, predictor, therm1, therm2, bestDOW, bestTIME)
    textEdit.move(60, 40)
    layout.addWidget(textEdit)

    layout.addSpacerItem(QtWidgets.QSpacerItem(20,0))

    feedback.setFont(QFont('Arial', 15))
    feedback.setText('''<html><head/><body><p style="line-height:50px; size: 20px; padding: 20px"><span>Type a potential post within the textarea on the left. Press ENTER to view live feedback</span></p> </body></html>''')
    feedback.setWordWrap(True)
    feedback.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,QtWidgets.QSizePolicy.MinimumExpanding))
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(feedback)
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)

## Organize the post score prediction thermometer


    therm1.setThermometer(0)

    miniOrganizer = QVBoxLayout()
    lab1 = QLabel()
    lab1.setText(" Post Score Prediction")
    lab1.setAlignment(Qt.AlignHCenter)
    lab1.setFont(QFont('Arial', 11))
    miniOrganizer.addWidget(lab1)
    miniOrganizer.addSpacerItem(QtWidgets.QSpacerItem(7,7))
    miniOrganizer.addLayout(therm1)
    miniOrganizer.setAlignment(Qt.AlignLeft)

## Organize the confidence theromemter

    therm2.setThermometer(0)

    confOrganizer = QVBoxLayout()
    confLabel = QLabel()
    confLabel.setText("Prediction Confidence")
    confLabel.setAlignment(Qt.AlignHCenter)
    confLabel.setFont(QFont('Arial', 11))
    confOrganizer.addWidget(confLabel)
    confOrganizer.addSpacerItem(QtWidgets.QSpacerItem(7,7))
    confOrganizer.addLayout(therm2)
    confOrganizer.setAlignment(Qt.AlignLeft)

## Organize the date thermometer

    dateOrganizer = QVBoxLayout()
    dateLabel = QLabel()
    dateLabel.setText("Best Submission Time")
    dateLabel.setAlignment(Qt.AlignHCenter)
    dateLabel.setFont(QFont('Arial', 11))
    dateOrganizer.addWidget(dateLabel)
    bestDateOrganizer = QHBoxLayout()

    bestDOW.setText("Mondays")
    bestDOW.setFont(QFont('Arial', 11,weight=QFont.Bold))
    atLabel = QLabel()
    atLabel.setText("at")
    atLabel.setFont(QFont('Arial', 11))

    bestTIME.setText("7:00 UTC")
    bestTIME.setFont(QFont('Arial', 11, weight=QFont.Bold))
    bestDateOrganizer.addWidget(bestDOW)
    bestDateOrganizer.addWidget(atLabel)
    bestDateOrganizer.addWidget(bestTIME)

    dateOrganizer.addLayout(bestDateOrganizer)
    dateOrganizer.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
    dateOrganizer.setAlignment(Qt.AlignCenter)

    topBar = QHBoxLayout()
    topBar.addSpacerItem(QtWidgets.QSpacerItem(7,0))

    topBar.addLayout(miniOrganizer)
    topBar.addSpacerItem(QtWidgets.QSpacerItem(20,0))
    topBar.addLayout(confOrganizer)
    topBar.addLayout(dateOrganizer)

    upperLayout.addSpacerItem(QtWidgets.QSpacerItem(0,20))
    upperLayout.addLayout(topBar)
    upperLayout.addSpacerItem(QtWidgets.QSpacerItem(0, 20))
    upperLayout.addLayout(layout)
    window.setGeometry(400,400,1200,600)
    window.setLayout(upperLayout)


    dummyPost = dict()
    dummyPost["title"] = ""
    dummyPost["selftext"] = ""
    dummyPost["created_utc"] = calendar.timegm(time.gmtime())
    predictor.predictPost(dummyPost)

    window.show()
    sys.exit(app.exec_())
