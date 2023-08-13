import nltk
import sys
import os
import string
from nltk.corpus import stopwords
from math import log
from operator import itemgetter

# Constants
FILE_MATCHES = 1
SENTENCE_MATCHES = 1

# Load NLTK stopwords
nltk.download("stopwords")
nltk.download("punkt")
nltk_stopwords = set(stopwords.words("english"))

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                files[filename] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    document = document.lower()
    tokens = nltk.word_tokenize(document)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in nltk_stopwords]
    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    total_documents = len(documents)
    all_words = set(word for words in documents.values() for word in words)
    
    for word in all_words:
        appearances = sum(1 for words in documents.values() if word in words)
        idf = log(total_documents / (1 + appearances))
        idfs[word] = idf
    
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = {}
    
    for filename, words in files.items():
        score = sum(idfs[word] for word in query if word in words)
        file_scores[filename] = score
    
    sorted_files = sorted(file_scores.items(), key=itemgetter(1), reverse=True)
    top_files = [filename for filename, _ in sorted_files[:n]]
    
    return top_files

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = {}
    
    for sentence, words in sentences.items():
        idf_score = sum(idfs[word] for word in query if word in words)
        query_density = sum(1 for word in words if word in query) / len(words)
        sentence_scores[sentence] = (idf_score, query_density)
    
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)
    top_sentences = [sentence for sentence, _ in sorted_sentences[:n]]
    
    return top_sentences


if __name__ == "__main__":
    main()
