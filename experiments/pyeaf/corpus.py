from pyeaf.pyeaf import EAFReader
from pyeaf.text import TextStemmer, RSLStemmer, Vocabulary, GramBinarizer


class CorpusData:
    def __init__(self, phrase_border=True):
        self.phrase_border = phrase_border
        self.corpus = None
        self.status = False

        pass

    def build(self, corpus):
        self.corpus = corpus

    pass
