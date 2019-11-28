import re

from ml_5.external import PorterStemmer
from ml_5.util import convert_to_features
from util.file.data_loader import read_file
from util.logger import LoggerBuilder

logger = LoggerBuilder().with_name("text_processing").build()


def is_spam(name, file_path, model, vocabulary, vocabulary_size):
    email = process_text(read_file(file_path), vocabulary)
    email_features = convert_to_features(email, vocabulary_size)
    logger.info('%s is %s', name, 'spam' if model.predict(email_features) == 1 else 'not spam')


def process_text(content, vocabulary):
    content = content.lower()
    content = re.compile('<[^<>]+>').sub(' ', content)
    content = re.compile('[0-9]+').sub(' number ', content)
    content = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', content)
    content = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', content)
    content = re.compile('[$]+').sub(' dollar ', content)
    content = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', content)
    content = [word for word in content if len(word) > 0]

    # Stem the email contents word by word
    stemmer = PorterStemmer()
    processed_content = []
    word_indices = []
    for word in content:
        # Remove any remaining non alphanumeric characters in word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = stemmer.stem(word)

        if len(word) < 1:
            continue

        processed_content.append(word)

        try:
            word_indices.append(vocabulary.index(word))
        except ValueError:
            pass

    return word_indices