from nltk.corpus import stopwords # Provides list of words to remove from emails.
from nltk.stem.wordnet import WordNetLemmatizer # Words to neutral from (nouns by default) eg kills & killing > kill.
import string # Allows string operations.
from nltk import word_tokenize # Tokenize a string to split off punctuation.
import gensim # Topic modelling, document indexing and similarity retrieval.
from random import randint # Allows generation of a random integer.
import os # The OS.
from collections import Counter # Allows counting of hashable objects.
from nltk import NaiveBayesClassifier, classify # We can classify arbitrary text.
import pickle # Converts python objects into a storable format and vice versa.
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier


def main():

    # Calls the "extract_csv_data" function to extract email data.
    print("\n>extracting email data..\n")

    # Specify the folders containing email data.
    spam = extract_data('enron-data-set/enron1/spam/')
    ham = extract_data('enron-data-set/enron1/ham/')

    # The emails are laid out in a format of 2 columns.
    # Column 1 is the email content.
    # Column 2 is whether the email is spam or ham.
    #
    # The next two lines iterate through all the emails.
    # Any email with spam in column 2 is added to spam_emails.
    # Any email with ham in column 2 is added to ham_emails.
    #
    # The entire collection of emails is then stored in all_emails by adding spam_emails and ham_emails.
    spam_emails = [(email, 'spam') for email in spam]
    ham_emails = [(email, 'ham') for email in ham]
    all_emails = spam_emails + ham_emails

    # Random number used to index a sample email later.
    random_email = randint(1, 500)

    # Print the amount of emails we'll be working with.
    # Print number of unique emails, followed by the randint email as a sample.
    print('There are a total of {} non-duplicate emails.\n'.format(len(all_emails)))
    print('Sample email, unstructured content:\n\n', all_emails[random_email])

    # Set the first 20000 emails as training set, the rest as testing set.
    # Send these emails to the "clean_data" function. This is a long process.
    print("\n>removing unecessary words..\n")
    cleaned_enron = clean_data(all_emails)

    # Split the cleaned enron dataset into a training and testing set.
    # When not testing set [0:1000] to [0:20000] and [1001:2000] to [20000:].
    training_set = cleaned_enron[0:20000]
    testing_set = cleaned_enron[20000:]

    # Print the randint email in the training set after being normalized.
    print('Sample email, normalized content:\n\n', training_set[random_email])

    # Call the create_dictionary function to gererate a dictionary from the training set.
    # This gets stored in created_dictionary.
    created_dictionary = create_dictionary(training_set)

    # Send the training and testing set to the process_dataset function along with the dictionary.
    process_dataset(training_set, testing_set, created_dictionary)

    # Call the get_features function, sending each email one at a time.
    # The all_features variable will contain tokens in a bag-of-words format.
    all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]

    # Call the train function, sending all_features and a value to represent the % split between training and testing.
    # The variables below will contain values for the train_set, test_set and classifier.
    train_set, test_set, classifier = train(all_features, 0.8)

    # Send the train_set, test_set and the classifier to the evaluate function.
    evaluate(train_set, test_set, classifier)

    # Display a result that will contain the top *number specified* words that appear in ham or spam emails.
    classifier.show_most_informative_features(20)

    # The generated classifer gets saved to a pickle file, so it can be used in future on other email sets.
    save_classifier = open("naivebayes.pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


    # CLASSIFIERS
    # Print additional classifiers accuracy results when ran against the test_set.
    mnb_classifier = SklearnClassifier(MultinomialNB())
    mnb_classifier.train(train_set)
    print("MNB_classifier accuracy percent:", (classify.accuracy(mnb_classifier, test_set)) * 100)

    bernouliNB_classifier = SklearnClassifier(BernoulliNB())
    bernouliNB_classifier.train(train_set)
    print("BernoulliNB_classifier accuracy percent:", (classify.accuracy(bernouliNB_classifier, test_set)) * 100)

    logisticRegression_classifier = SklearnClassifier(LogisticRegression())
    logisticRegression_classifier.train(train_set)
    print("LogisticRegression_classifier accuracy percent:", (classify.accuracy(logisticRegression_classifier, test_set)) * 100)

    svc_classifier = SklearnClassifier(SVC())
    svc_classifier.train(train_set)
    print("SVC_classifier accuracy percent:", (classify.accuracy(svc_classifier, test_set)) * 100)

    linearsvc_classifier = SklearnClassifier(LinearSVC())
    linearsvc_classifier.train(train_set)
    print("LinearSVC_classifier accuracy percent:", (classify.accuracy(linearsvc_classifier, test_set)) * 100)


# This function is to obtain the email contents from the enron folders.
def extract_data(folder):
    # This list starts empty so data can be added to it.
    a_list = []

    # The loop below will read each email in the specified folders and add the contents to a_list.
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r', encoding='latin-1')
        a_list.append(f.read())
    f.close()

    # Send the list back.
    return a_list


# Sends each doc to the "clean" function.
def clean_data(data):
    return [clean(doc).split(' ') for doc in data]


# Clean up the data, remove unwanteds words that are such as I, A etc.. (stopwords.word('english')).
def clean(emailbody):
    # The emails had to be converted to a string.
    emailbody = str(emailbody)

    # The below 3 lines set the variables to -
    # Exclude a list of english words
    # Remove the punctuation
    # Set the lemmatizer
    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    # Go through every word in the email, append any word we're keeping to word_free
    word_free = " ".join([i for i in emailbody.lower().split() if i not in words_to_exclude])

    # Go though all the characters in word_free, add letters and numbers to punc_free.
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)

    # Lemmatize every word and add it back to normalised.
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    # Send the cleaned text back.
    return normalized


# The function below is used to create the dictionary.
def create_dictionary(training_set):

    # Implements the concept of Dictionary – a mapping between words and their integer ids, using the training set.
    print("\n>generating dictionary..\n")
    dictionary = gensim.corpora.Dictionary(training_set)

    # Return the created dictionary.
    return dictionary


# The function below is pass the email data through TFIDF and LSI.
def process_dataset(training_set, testing_set, dictionary):

    # Keep tokens that are contained in at least 20 documents (absolute number).
    # Keep tokens that are contained in no more than 0.1 documents (fraction of total corpus size, not absolute).
    # 0.1 being 10% of the Corpus, a word occurs in more than 10% it will be too common a term to warrant classifying.
    print("\n>filtering dictionary..\n")
    dictionary.filter_extremes(no_below=20, no_above=0.1)

    # Converts each doc into the bag-of-words format, list of (token_id, token_count).
    print("\n>generating training matrix..\n")
    matrix = [dictionary.doc2bow(doc) for doc in training_set]

    # TFIDF is a numerical statistic that aims to reflect the importance of a word within a document.
    # Output our tfidf_model: "TfidfModel(num_docs=200000, num_nnz=16017590)" where num_nnz is number of non-zero elements.
    tfidf_model = gensim.models.TfidfModel(matrix, id2word=dictionary)

    # LSI aims to identify patterns in relationships between terms and concepts in unstructured text.
    # Output our lsi_model: "LsiModel(num_terms=47483, num_topics=100, decay=1.0, chunksize=20000)
    lsi_model = gensim.models.LsiModel(tfidf_model[matrix], id2word=dictionary, num_topics=100)

    # Generate 100 topics containing 10 words each.
    topics = lsi_model.print_topics(num_topics=100, num_words=8)

    # Print out each topic generated by the lsi_model.
    for topic in topics:

        print(topic)

    # Below is repeating lines ### - ###, but applied to the testing dataset of emails.
    # Please note that the dictionary used here has been previously generated from the training set.
    # This code will be cleaned eventually.
    print("\n>generating testing matrix..\n")
    matrix = [dictionary.doc2bow(doc) for doc in testing_set]
    tfidf_model = gensim.models.TfidfModel(matrix, id2word=dictionary)
    lsi_model = gensim.models.LsiModel(tfidf_model[matrix], id2word=dictionary, num_topics=100)
    topics = lsi_model.print_topics(num_topics=100, num_words=10)

    # Print one topic at a time.
    for topic in topics:

        print(topic)


# Yes the following two functions seem to repeat what has already been done, but its required to work.
def get_features(text, setting):

    # Set the stoplist to english.
    stoplist = stopwords.words('english')

    # We pass the 'bow' (bag of words) setting. This will return words along with their count of occurances.
    # The words will be not be returned if they appear in stopwords.
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}


# This again, is a function to convert words in the tokens to a lemmatized format.
def preprocess(sentence):

    # Return lemmatized words, converted to lowercase.
    lemma = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    return [lemma.lemmatize(word.lower()) for word in tokens]


# The function will receive all_features and a percentage to split the words into a training and testing set.
def train(features, samples_proportion):

    # Calculate the amount of words to be placed in the training and testing set.
    train_size = int(len(features) * samples_proportion)

    # The train_set variable will contain words from zero up to the percentage of amount specified.
    # The test_set variable will contain words from onwards from the amount of percentage specified.
    # For example, 100 words, 80% split:
    # train_set will contain words from 0 to 80.
    # test_set will contain words from 80 to 100.
    train_set, test_set = features[:train_size], features[train_size:]

    # Print the amount of emails in the training set and testing set. For user informative purposes.
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')

    # Feed the train_set of words into the NaiveBayers classifier, whatever it learns is stored in the classifier variable.
    classifier = NaiveBayesClassifier.train(train_set)

    # Send back the train_set, test_set and the newly learnt classifier.
    return train_set, test_set, classifier


# This function will print the accuracy of the classifier when run against the train and test set.
def evaluate(train_set, test_set, classifier):

    # Prints accuracy information for user informative purposes.
    print ('Accuracy of the Naives Bayes Classifier on the training set: ', (classify.accuracy(classifier, train_set)) * 100)
    print ('Accuracy of the Naives Bayes Classifier on the test set: ',(classify.accuracy(classifier, test_set)) * 100)


main()
