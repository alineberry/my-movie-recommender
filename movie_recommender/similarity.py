from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd

class SimilarityPredictions(object):
    '''This class calculates a similarity matrix from latent embeddings.
    There is a method to save this similarity model locally, and a method for
    predicting similar items from the matrix.
    Input: embeddings - a pandas dataframe of items and latent dimensions.
            similarity_metric = str definining the similarity metrics to use'''

    def __init__(self, embeddings, similarity_metric='cosine'):
        assert similarity_metric in ['cosine'], "unsupported similarity metric."
        self.embeddings = embeddings
        self.ids = embeddings.index.tolist()
        if similarity_metric == 'cosine':
            self.similarity_matrix = self.calculate_cosine_similarity_matrix()

    def calculate_cosine_similarity_matrix(self):
        '''Calculates a cosine similarity matrix from the embeddings'''
        similarity_matrix = pd.DataFrame(cosine_similarity(
            X=self.embeddings),
            index=self.ids)
        similarity_matrix.columns = self.ids
        return similarity_matrix

    def calculate_euclidean_distances_matrix(self):
        '''Calculates a cosine similarity matrix from the embeddings'''
        euclisimilarity_matrixdean_matrix = pd.DataFrame(euclidean_distances(
            X=self.embeddings),
            index=self.ids)
        similarity_matrix.columns = self.ids
        return euclsimilarity_matrixidean_matrix

    def save_similarity_model(self, path, file_format='csv'):
        '''Save similarity matrix locally'''
        assert file_format in ['csv', 'pickle'], "unsupported format"
        if file_format == 'csv':
            self.similarity_matrix.to_csv(path, header=True, index=True)
        elif forfile_formatmat == 'pickle':
            self.similarity_matrix.to_pickle(path)

    def predict_similar_items(self, seed_item, n):
        '''Use the similarity_matrix to return n most similar items.
        '''
        similar_items = pd.DataFrame(self.similarity_matrix.loc[seed_item])
        similar_items.columns = ["similarity_score"]
        similar_items = similar_items.sort_values('similarity_score', ascending=False)
        similar_items = similar_items.head(n)
        similar_items.reset_index(inplace=True)
        similar_items = similar_items.rename(index=str, columns={"index": "item_id"})
        return similar_items.to_dict()
