from sklearn.decomposition import TruncatedSVD
import pandas as pd


class CollaborativeFiltering(object):
    '''Methods for performing Collaborative Filtering.
    Input:
    data - A pandas dataframe of user and items interactions.
           Users are rows and items are columns.
    '''

    def __init__(self, data):
        self.data = data
        self.users = data.index.tolist()
        self.items = data.columns.tolist()

    def get_svd_embeddings(self, n):
        '''Compress the original user-item interaction matrix into n latent features
        using matrix factorization.
        Returns:
        item_latent_df - a matrix for mapping items into latent space.
        user_latent_df - a matrix for mapping users into latent space.'''
        item_latent_df = pd.DataFrame([])
        user_latent_df = pd.DataFrame([])
        return item_latent_df, user_latent_df

    def get_neural_network_embeddings(self, n):
        '''Compress the original user-item interaction matrix into n latent features
        using a neural network approach to dimensionality reduction.
        Returns:
        item_latent_df - a matrix for mapping items into latent space.
        user_latent_df - a matrix for mapping users into latent space.'''
        item_latent_df = pd.DataFrame([])
        user_latent_df = pd.DataFrame([])
        return item_latent_df, user_latent_df
