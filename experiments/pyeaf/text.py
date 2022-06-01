import pymystem3
import re
from typing import List, Union
import numpy as np


class TextStemmer(object):

    def __init__(self):
        self.stemmer = pymystem3.Mystem(disambiguation=False, entire_input=False)
        self.relex = re.compile(r'[,()=|]')
        self.retext = re.compile(r'[^-а-яa-z0-9_]', re.I)

    def _stem_gram(self, text: str):
        text = self.retext.sub(' ', text)

        def f(lem):
            if lem['analysis']:
                lem = lem['analysis'][0]
                lex = lem['lex']
                gr = self.relex.sub(' ', lem['gr']).split()
            else:
                lex = lem['text']
                gr = ['']
            return {'lex': lex, 'gram': list(set(gr))}

        text = self.stemmer.analyze(text)
        text = map(f, filter(lambda x: 'analysis' in x, text))

        return list(text)

    def _stem_simple(self, text: str):
        text = self.retext.sub(' ', text)
        text = self.stemmer.lemmatize(text)
        return [{'lex': t, 'gram': []} for t in text]

    def stem(self, text: Union[str, List[str]], gram=True):
        if isinstance(text, str):
            text = [text]

        f = self._stem_simple if not gram else self._stem_gram
        text = [f(t) for t in text]
        lex = [[t['lex'] for t in sentence] for sentence in text]
        gr = [[t['gram'] for t in sentence] for sentence in text]
        return lex, gr

    def __delete__(self, instance):
        self.stemmer.close()


class RSLStemmer:
    redact = re.compile(r'[a-zа-я]*(-[a-zа-я]+){2,}')
    repspl = re.compile(r'(\d(?:ps|pl)):([-а-яa-z0-9]+):(\d(?:ps|pl))', re.I)
    restwr = re.compile(r'prtcl')
    renums = re.compile(r'\b\d[\d\s]+\b')
    reclf = re.compile(r'(clf)((:|\.)[a-zа-я]+)+', re.I)

    @classmethod
    def stem_sentence(cls, text: Union[str, List[str]]):
		# сюда можно передавать и строку и список из предложений
        text = " ".join(text) if hasattr(text, '__iter__') else text
        text = text.lower()
        text = cls.redact.sub('<dact>', text)  # точно не нужно, ruT5 умеет предсказывать дактиль 
        text = cls.repspl.sub(r'\1 \2 \3', text)  # разбить пробелом штуки которые были через двоеточие
        text = cls.renums.sub(' <nums> ', text)  # тоже кажется необязательно
        text = cls.restwr.sub(' ', text)
        # нельзя путать куски классификатора с синонимичными жестами, они по-разному выглядят
        # разметка вот такая должна быть CLF:плоский.предмет:держать.в.карман
        sent = []
        for word in text.split():
            if cls.reclf.match(word):
                clf = re.split(r':|\.', cls.reclf.match(word).group(0))
                for k, el in enumerate(clf[1:]):
                    clf[k+1] = 'clf.' + el
                sent += clf
            else:
                sent += [word]
        text = ' '.join(sent)
        
        text = text.replace(':', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.split()

class GramBinarizer:
    def __init__(self, phrase_border=False):
        self.dict = None
        self.grams = set()
        self.num_grams = 0
        self.phrase_border = phrase_border

    def fit(self, corpus):
        for sentence in corpus:
            for word_grams in sentence:
                self.grams |= set(word_grams)

        self.grams = list(self.grams)
        self.num_grams = len(self.grams)

        self.dict = {gr: ind for ind, gr in enumerate(self.grams)}
        return self

    def transform(self, corpus):
        bin_corpus = []
        for sentence in corpus:
            bin_sentence = []
            sentence = [[]] + sentence + [[]] if self.phrase_border else sentence
            for word_grams in sentence:
                bins = np.zeros(self.num_grams)
                index = [self.dict[gr] for gr in word_grams if gr in self.dict]
                bins[index] = 1.
                bin_sentence.append(bins)
            bin_corpus.append(bin_sentence)
        return bin_corpus


class VocabularyVectorizer(object):
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    MASK = "<mask>"
    EMP = ""

    def __init__(self, phrase_border=False, sentence_len=0, vectorize=True):
        self.phrase_border = phrase_border
        self.vocab_s = {}
        self.vocab_i = []
        self.word_count = 0
        self.max_len = 0
        self.sentence_len = sentence_len
        self.vectorize = vectorize

        self._add_phrase([self.MASK, self.UNK, self.BOS, self.EOS, self.EMP])
        if self.phrase_border:
            self.sentence_len = self.sentence_len + 2

        self.mask_ind = self.vocab_s[self.MASK]
        self.bos_ind = self.vocab_s[self.BOS]

    def _add_word(self, word: str):
        if not (word in self.vocab_s):
            self.vocab_i.append(word)
            self.word_count = len(self.vocab_i)
            self.vocab_s.update({word: self.word_count - 1})

    def _add_phrase(self, phrase: List[str]):
        length = len(phrase) + 2 if self.phrase_border else len(phrase)
        self.max_len = max(self.max_len, length, self.sentence_len)
        for word in phrase:
            self._add_word(word)

    def _add_text(self, text: List[List[str]]):
        for phrase in text:
            self._add_phrase(phrase)

    def _get_word(self, ind: int):
        return self.vocab_i[ind] if ind < self.word_count else self.UNK

    def _get_phrase(self, inds: List[int]):
        return [self._get_word(ind) for ind in inds]

    def _get_text(self, text: List[List[int]]):
        return [self._get_phrase(inds) for inds in text]

    def _ind_word(self, word: str):
        return self.vocab_s.get(word, self.vocab_s[self.UNK])

    def _ind_phrase(self, phrase: List[str]):
        if self.vectorize:
            ind_phrase = np.array([self.mask_ind]*self.max_len)
            phrase = [self.BOS] + phrase + [self.EOS] if self.phrase_border else phrase
            ind_phrase[:len(phrase)] = [self._ind_word(word) for word in phrase]
        else:
            phrase = [self.BOS] + phrase + [self.EOS] if self.phrase_border else phrase
            ind_phrase = [self._ind_word(word) for word in phrase]
        return ind_phrase

    def _ind_text(self, text: List[List[str]]):
        return [self._ind_phrase(phrase) for phrase in text]

    def fit(self, text: List[List[str]]):
        self._add_text(text)
        return self

    def text_to_index(self, text: List[List[str]]):
        return self._ind_text(text)

    def index_to_text(self, text):
        return self._get_text(text)
