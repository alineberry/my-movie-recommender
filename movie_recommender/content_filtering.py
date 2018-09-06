from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd


class ContentFiltering(object):
    '''Methods for performing Content Filtering.
    Input: data - a pandas dataframe of items and features.
        similarity_metric = str definining the similarity metrics to use.'''

    def __init__(self, data):
        self.data = data
        self.ids = data.index.tolist()

    def tfidf_tokenizer(self, min_df, ngram_range, documents_column_name):
        '''Performes TF-IDF tokenization. Use documents_column_name to specify the
        column containing the text data to be tokenized.
        Returns a dataframe of TF-IDF features indexed by item id.'''
        tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data[documents_column_name])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=self.ids)
        return tfidf_df

    def get_svd_embeddings(self, feature_matrix, n):
        '''Compress the original feature matrix into n latent features using matrix factorization.
        Returns a dataframe with n latent features.'''
        svd = TruncatedSVD(n_components=n)
        latent_matrix = svd.fit_transform(feature_matrix)
        latent_df = pd.DataFrame(latent_matrix, index=self.ids)
        return latent_df

    def save_embeddings(self, latent_df, path, file_format='csv'):
        '''Save embeddings locally'''
        assert file_format in ['csv', 'pickle'], "unsupported format"
        if file_format == 'csv':
            latent_df.to_csv(path, header=True, index=True)
        elif forfile_formatmat == 'pickle':
            latent_df.to_pickle(path)
