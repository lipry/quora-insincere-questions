import re
import operator

import math

import nltk
import numpy as np
from pattern.text import Sentence
from pattern.text.en import parse, modality, sentiment, mood
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

from tqdm.auto import tqdm
import spacy
tqdm.pandas()


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    covered_word_count = 0
    oov_word_count = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            covered_word_count += vocab[word]
        except KeyError:

            oov[word] = vocab[word]
            oov_word_count += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(covered_word_count /
                                                            (covered_word_count + oov_word_count)))
    return sorted(oov.items(), key=operator.itemgetter(1))[::-1]


def build_dictionary(questions):
    d = {}
    for sentence in tqdm(questions):
        for word in sentence:
            try:
                d[word] += 1
            except KeyError:
                d[word] = 1
    return d


def cleaning_questions(df, column='question_text'):
    cleaned_questions = df[column]\
        .progress_apply(lambda x: clean_text(x))\
        .progress_apply(lambda x: correct_mispelling(x))\
        .progress_apply(lambda x: clean_numbers(x))\
        .progress_apply(lambda x: word_tokenize(x))

    return cleaned_questions


def idf_dictionary_builder(documents):
    vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    X = vectorizer.fit_transform(documents)
    idf_d = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    return vectorizer.vocabulary_, X, idf_d


nlp = spacy.load('en_core_web_sm')
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')


def ner_replacer(x):
    return [number_entity_checker(token) for token in nlp(x)]


def number_entity_checker(token):
    return token.ent_type_ if token.ent_type_ in ['TIME', 'PERCENT', 'MONEY', 'CARDINAL', 'DATE'] else token.text


def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre': 'theater',
                'cancelled': 'canceled',
                'labour': 'labor',
                'organisation': 'organization',
                'wwii': 'world war 2',
                'citicise': 'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'}

mispellings, mispellings_re = _get_mispell(mispell_dict)


def correct_mispelling(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


to_remove = ['a','to','of','and', ' ', '  ', '…']


def remove_stop_words(x, stop_words=to_remove):
    return [word for word in x if word not in stop_words]


def clean_stopwords(questions, word_list, th):
    stop_w = get_stop_words(word_list, threshold=th)
    return [remove_stop_words(sentence, stop_words=stop_w) for sentence in tqdm(questions)]


def sentiment_pattern(text):
    sent = sentiment(text)
    return sent[0], sent[1]


def get_modality_mood(text):
    t = parse(text, lemmata=True)
    t = Sentence(t)
    return modality(t), mood(t)


def get_specificity(vocab_corpus_A, vocab_corpus_B):
    total_A = sum([v for v in vocab_corpus_A.values()])
    total_B = sum([v for v in vocab_corpus_B.values()])
    freqA = dict((key, value/total_A) for key, value in vocab_corpus_A.items())
    freqB = dict((key, value/total_B) for key, value in vocab_corpus_B.items())
    return {token: (f - freqA[token]) / math.sqrt(freqA[token]) for token, f in freqB.items()}


def specificity_average(question, specificity_dict):
    l = [specificity_dict[term] if term in specificity_dict else 0.0 for term in question]
    return np.mean(l) if len(l) > 0 else 0.0


def get_stop_words(word_dict, threshold=0.0):
    return {word: v for word, v in word_dict.items() if v < threshold}.keys()


def POSTagger(t):
    return [tag[1] for tag in nltk.pos_tag(t)]


