import re
import spacy


class Atomizer:
    _TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
    _ALL_DIGITS_PATTERN = re.compile('\d+')
    _spacy_nlp = spacy.load('en')

    def __init__(self, ngram_range=(1, 1), stop_words=None, boost_terms=None, map_terms=None, preprocessor=None):
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.boost_terms = boost_terms
        self.map_terms = map_terms  # TODO not implemented
        self.preprocessor = preprocessor

    def atomize(self, raw_document):
        ngrams = self._build_ngrams(self._tokenize(self._preprocess(raw_document)))
        if self.boost_terms is not None:
            return self._boost(ngrams)
        else:
            return ngrams

    @classmethod
    def normalize_token(cls, token):
        if token.startswith('_'):
            return token
        else:
            return cls._lemmatize(token.lower())

    @classmethod
    def normalize_multitoken(cls, multitoken):
        return ' '.join([
            cls.normalize_token(token)
            for token in multitoken.split(' ')
        ])

    @classmethod
    def _lemmatize(cls, token):
        return cls._spacy_nlp(unicode(token))[0].lemma_

    def _preprocess(self, document):
        if self.preprocessor is not None:
            return self.preprocessor(document)
        else:
            return document

    def _tokenize(self, document):
        return [self.normalize_token(token)
                for token
                in self._TOKEN_PATTERN.findall(document)
                if self._accept_token(token)]

    def _boost(self, stringified_ngrams):
        boosted_ngrams = []

        # upweighting: counting a term as if it occurred multiple times
        for ngram in stringified_ngrams:
            for i in xrange(0, self.boost_terms.get(ngram, 1)):
                boosted_ngrams.append(ngram)

        return boosted_ngrams

    def _accept_token(self, token):

        # reject too short tokens
        if len(token) == 1: return False

        # reject standalone numbers
        if self._ALL_DIGITS_PATTERN.match(token): return False

        return True

    def _accept_ngram(self, ngram):

        # drop n-grams containing stop words
        if '-STOP-' in ngram: return False

        # drop n-grams containing pronouns
        if '-PRON-' in ngram: return False

        # drop "doubles"
        if len(ngram) == 2:
            if ngram[0] == ngram[1]: return False

        return True

    def _build_ngrams(self, tokens):
        """
        Turn tokens into a sequence of n-grams
        """

        # handle stop words
        if self.stop_words is not None:
            tokens = [w
                      if w not in self.stop_words
                      else '-STOP-'
                      for w in tokens]

        stringified_ngrams = []

        min_n, max_n = self.ngram_range
        if max_n == 1:
            # handle unigrams
            for token in tokens:
                if self._accept_ngram([token]):
                    stringified_ngrams.append(token)
        else:
            # handle n-grams
            n_tokens = len(tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_tokens + 1)):
                for i in xrange(n_tokens - n + 1):
                    ngram = tokens[i: i + n]
                    if self._accept_ngram(ngram):
                        stringified_ngrams.append(' '.join(ngram))

        return stringified_ngrams
