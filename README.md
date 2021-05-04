# Getting Started

This final project requires the following libraries (all of which should be included in the anaconda distribution):

 - [ ] PyQt5
 - [ ] Tensorflow
 - [ ] Keras
 - [ ] Numpy
 - [ ] NLTK
 - [ ] Pickle

You also need to download the proper NLTK corpus sets. 

You need
 - [ ] stopwords
 - [ ] vader_lexicon
 - [ ] punkt
 - [ ] wordnet

To install them, simply execute
`python -c "import nltk;nltk.download('stopwords');
nltk.download('vader_lexicon');
nltk.download('punkt');
nltk.download('wordnet');"`
 
# Running the GUI

To use the pre-scraped data included within the repository for r/Conservative and use the pre-trained model, simply execute the command

`python gui.py`

![GUI screenshot](https://i.ibb.co/G7WRrkZ/GUI.png)

For evaluation purposes, everything that is needed is included within this command. Do not run any further commands, unless you wish to use the API.

**Be patient!** The GUI may take up to 5 minutes to start.

# Building the Corpus

Before a model can be trained and used, reddit posts must be scraped from pushshift.io. A prebuilt corpus for r/Conservative is included within the repository, as well as a pretrained model. To continue to scrape from this subreddit, or to scrape from a new subreddit, run 

    python scrape.py <subreddit name> <number of posts>


Posts will be scraped by rounding up to the nearest 1000.

# Training the Model

Once posts have been scraped from a particular subreddit, you may train a deep neural network to begin making predictions from that subreddit. Simply run

    python train.py <subreddit name>

to get started training the model. The finished model will be automatically saved to `model > best` using the built-in model serialization with Keras. Likewise, the exported vocabulary and other necessary data attributes for feature extraction will be saved to `model` 

**Therefore, training a model on one subreddit will automatically overwrite training performed on a previous subreddit**

