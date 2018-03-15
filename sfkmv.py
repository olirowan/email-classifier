import numpy as np # Linear algebra.
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv).
import re
from nltk.corpus import stopwords # Provides list of words to remove from emails.
from nltk.stem.wordnet import WordNetLemmatizer # Words to neutral from (nouns by default) eg kills & killing > kill.
import string
import gensim # Topic modelling, document indexing and similarity retrieval.


def main():

    # Calls the "extract_csv_data" function to extract email data.
    print("\n>extracting email data..\n")
    emails = extract_csv_data(
        'emails.csv',
        ['file'],
        [['notes_inbox', 'discussion_threads']]
    )

    # Sends the email bodies to the "remove_duplicates" function.
    email_bodies = emails.message.as_matrix()
    print("\n>removing duplicates..\n")
    unique_emails = remove_duplicates(email_bodies)

    # Print number of unique emails, followed by 1000th email as a sample.
    print('There are a total of {} non-duplicate emails.\n'.format(len(unique_emails)))
    print('Sample email, unstructured content:\n\n', unique_emails[1000])

    # Set the first 200000 emails as training set, the rest as testing set.
    # Send these emails to the "clean_data" function. This is a long process.
    print("\n>removing unecessary words..\n")
    training_set = clean_data(unique_emails[0:200000])
    testing_set = clean_data(unique_emails[200000:]) #currently unused

    # Print the 1000th email in the training set after being normalized.
    print('Sample email, normalized content:\n\n', training_set[1000])

    # Implements the concept of Dictionary – a mapping between words and their integer ids, using the training set.
    print("\n>generating dictionary..\n")
    dictionary = gensim.corpora.Dictionary(training_set)

    # Keep tokens that are contained in at least 20 documents (absolute number).
    # Keep tokens that are contained in no more than 0.1 documents (fraction of total corpus size, not absolute).
    print("\n>filtering dictionary..\n")
    dictionary.filter_extremes(no_below=20, no_above=0.1)
    print(dictionary)

    # Converts each doc into the bag-of-words format, list of (token_id, token_count).
    print("\n>generating matrix..\n")
    matrix = [dictionary.doc2bow(doc) for doc in training_set]

    tfidf_model = gensim.models.TfidfModel(matrix, id2word=dictionary)
    lsi_model = gensim.models.LsiModel(tfidf_model[matrix], id2word=dictionary, num_topics=100)
    topics = lsi_model.print_topics(num_topics=100, num_words=10)

    for topic in topics:

        print(topic)


# Explain this.
def extract_csv_data(file, cols_to_clean = [], exclude = [[]]):

    data = pd.read_csv(file)

    for i, col in enumerate(cols_to_clean):
        exclude_pattern = re.compile('|'.join(exclude[i]))
        data = data[data[col].str.contains(exclude_pattern) == False]

    return data


# Compares the emails in the CSV and removes duplicate emails.
def remove_duplicates(data):

    processed = set()
    result = []
    pattern = re.compile('X-FileName: .*')
    pattern2 = re.compile('X-FileName: .*?  ')

    for doc in data:
        doc = doc.replace('\n', ' ')
        doc = doc.replace(' .*?nsf', '')
        match = pattern.search(doc).group(0)
        match = re.sub(pattern2, '', match)

        if match not in processed:
            processed.add(match)
            result.append(match)

    return result


# Sends each doc to the "clean" function.
def clean_data(data):

    return [clean(doc).split(' ') for doc in data]


# Clean up the data, remove unwanteds words that are such as I, A etc.. (stopwords.word('english')).
def clean(doc):

    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized


main()