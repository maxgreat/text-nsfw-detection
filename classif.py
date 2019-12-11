from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer=.'word', binary=False, decode_error='strict',
        dtype=<'numpy.int64'>, encoding='utf-8', input='content',
        lowercase=True, max_df=1.0, max_features=None, min_df=1,
        ngram_range=(1, 1), preprocessor=None, stop_words=None,
        strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
        tokenizer=None, vocabulary=None)

