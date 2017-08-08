import pandas as pd
import numpy as np
import os
import datetime
import pickle


class MovieFilter(object):

    def __init__(self, movies, title_col='title'):
        self.movies = movies
        self.title_col = title_col

    def filter_string_length(self, length=60):
        title_lens = self.movies.apply(lambda x: len(x[self.title_col]), axis=1)
        mask = title_lens < length
        len0 = len(self.movies)
        self.movies = self.movies[mask]
        len1 = len(self.movies)
        self.print_filter_results('filter_string_length', len0, len1)

    def filter_english_words(self, num_allow=2):
        import string
        cwd = os.getcwd()
        words_en = set(line.strip() for line in open(os.path.join(cwd, "..", "data", "wordsEn.txt")))

        def check_if_english(row):
            title = row[self.title_col].lower()
            title = ''.join(c for c in title if c not in set(string.punctuation)).strip()
            title = ''.join(c for c in title if not c.isdigit()).strip()
            words = title.split(' ')
            count = 0
            for word in words:
                if word.strip() not in words_en:
                    count += 1
                    if count >= num_allow:
                        return False
            return True
        mask = self.movies.apply(check_if_english, axis=1)
        len0 = len(self.movies)
        self.movies = self.movies[mask]
        len1 = len(self.movies)
        self.print_filter_results('filter_english_words', len0, len1)

    def filter_release_year(self, min_year=1990):
        import re

        def get_release_year(row):
            title = row[self.title_col]
            year = re.search(r'\(\d{4}\)', title)
            if year:
                year = year.group(0)
            else:
                return None
            year = int(year.replace('(', '').replace(')', ''))
            return year
        release_year = self.movies.apply(get_release_year, axis=1)
        mask = release_year > min_year
        len0 = len(self.movies)
        self.movies = self.movies[mask]
        len1 = len(self.movies)
        self.print_filter_results('filter_release_year', len0, len1)

    def filter_rating_freq(self, threshold=200, movieId_col='movieId'):
        cwd = os.getcwd()
        ratings = pd.read_csv(os.path.join(cwd, "..", "data", "ratings.csv"))
        freq = ratings[movieId_col].value_counts()
        red_freq = freq[freq >= threshold]
        red_freq = red_freq.index.tolist()
        mask = self.movies[movieId_col].isin(red_freq)
        len0 = len(self.movies)
        self.movies = self.movies[mask]
        len1 = len(self.movies)
        self.print_filter_results('filter_rating_freq', len0, len1)

    @staticmethod
    def print_filter_results(filter_name, len0, len1):
        print('{} filtered out {} movies. Num before: {}. Num after: {}'.format(filter_name, len0 - len1, len0, len1))

    def reduce_ratings_dataset(self, ratings, movieId_col='movieId'):
        mask = ratings[movieId_col].isin(self.movies[movieId_col])
        len0 = len(ratings)
        ratings = ratings[mask]
        len1 = len(ratings)
        print('Filtered out {} ratings. Num before: {}. Num after: {}'.format(len0 - len1, len0, len1))


class PMF(object):
    
    def __init__(self, normalize_ids=True, rank=5, lamd=2, sig2=1/10., num_iter=50, verbose=False):
        self.norm_ids = normalize_ids
        self.d = rank
        self.lamd = lamd
        self.sig2 = sig2
        self.num_iter = num_iter
        self.preprocessed = False
        self.verbose = verbose

    def depersist_preprocessing_data(self):
        """
        Use this function to depersist the preprocessed data structures ratings(with normalized ids), omega, omega_u, 
        omega_v, and M
        :return: None if any of the files are not found. Array 'ratings' is returned if all files are found 
        """
        # check that all files exist in cwd first
        files = ['ratings_preprocessed.csv', 'omega.pkl', 'omega_u.pkl', 'omega_v.pkl', 'M.pkl']
        f_exists = [os.path.isfile(os.path.join(os.getcwd(), f)) for f in files]
        if False in f_exists:
            print('Some preprocessing data is not found:')
            print(zip(files, f_exists))
            return None
        ratings = np.genfromtxt(os.path.join(os.getcwd(), "ratings_preprocessed.csv"), delimiter=',')
        self.omega = pickle.load(open('omega.pkl', 'rb'))
        self.omega_u = pickle.load(open('omega_u.pkl', 'rb'))
        self.omega_v = pickle.load(open('omega_v.pkl', 'rb'))
        self.M = pickle.load(open('M.pkl', 'rb'))
        self.preprocessed = True
        return ratings
    
    def normalize_ids(self, ratings, normalize_userids=True, normalize_objids=True):
        """
        The mapping variables are 2-column numpy arrays. The first column contains the original
        ids and the second column contains the normalized ids they are mapped to
        """
        if normalize_userids:
            print('Normalizing user ids | {}'.format(datetime.datetime.now()))
            self.user_mapping = np.stack((np.unique(ratings[:,0]), np.arange(self.N1)), axis=1)
            self.user_mapping = self.user_mapping.astype(np.int64)
            # iterate through users
            for i in range(self.N1):
                if ((i+1) % 1000 == 0) and self.verbose:
                    print('iteration {} of {}'.format(i+1, self.N1))
                if self.user_mapping[i,0] != self.user_mapping[i,1]:
                    mask = ratings[:,0] == self.user_mapping[i,0]
                    ratings[mask,0] = self.user_mapping[i,1]
            print('saving user_mapping.csv to cwd')
            np.savetxt('user_mapping.csv', self.user_mapping, delimiter=',')
        if normalize_objids:
            print('Normalizing object ids | {}'.format(datetime.datetime.now()))
            self.obj_mapping = np.stack((np.unique(ratings[:,1]), np.arange(self.N2)), axis=1)
            self.obj_mapping = self.obj_mapping.astype(np.int64)
            # iterate through objects
            for j in range(self.N2):
                if ((j+1) % 1000 == 0) and self.verbose:
                    print('iteration {} of {}'.format(j+1, self.N2))
                if self.obj_mapping[j,0] != self.obj_mapping[j,1]:
                    mask = ratings[:,1] == self.obj_mapping[j,0]
                    ratings[mask,1] = self.obj_mapping[j,1]
            print('saving obj_mapping.csv to cwd')
            np.savetxt('obj_mapping.csv', self.obj_mapping, delimiter=',')
        return ratings
    
    def check_ids(self, ratings):
        """
        Returns a tuple of booleans indicating whether the user and object ids are correctly normalized.
        The correct ids are a range of integers starting at 1.
        """
        user_check_val = (np.unique(ratings[:,0]) - np.arange(self.N1)).sum()
        obj_check_val = (np.unique(ratings[:,1]) - np.arange(self.N2)).sum()
        user_check = False if user_check_val != 0 else True
        obj_check = False if obj_check_val != 0 else True
        print('user ids normalized: {} | object ids normalized: {}'.format(user_check, obj_check))
        return user_check, obj_check

    def fit(self, ratings, force_refresh=True):

        ratings = ratings[['userId', 'movieId', 'rating']].as_matrix()

        if ratings.shape[1] != 3:
            print('ratings matrix is incorrect shape')
            return

        print('force_refresh is set to: ' + str(force_refresh))

        # sparse matrix shape
        self.N1 = len(np.unique(ratings[:,0]))
        self.N2 = len(np.unique(ratings[:,1]))

        if (not self.preprocessed) or force_refresh:
            # normalizing ids is expensive and can be time consuming. check to see if the ids are
            # already normalized correctly before doing anything
            if self.norm_ids:
                user_check, obj_check = self.check_ids(ratings)
                ratings = self.normalize_ids(ratings, not user_check, not obj_check)
            np.savetxt('ratings_preprocessed.csv', ratings, delimiter=',')
            # obtain metadata/ data structures
            self.omega = self.build_omega(ratings)
            pickle.dump(self.omega, open('omega.pkl', "wb"))
            self.omega_u = self.build_omega_u(ratings)
            pickle.dump(self.omega_u, open('omega_u.pkl', "wb"))
            self.omega_v = self.build_omega_v(ratings)
            pickle.dump(self.omega_v, open('omega_v.pkl', "wb"))
            self.M = self.build_M(ratings)
            pickle.dump(self.M, open('M.pkl', "wb"))
            self.preprocessed = True

        self.u, self.v, self.L = self.execute_training_epochs(ratings)
        
        np.savetxt('U.csv', self.u, delimiter=',')
        np.savetxt('V.csv', self.v, delimiter=',')
        np.savetxt('objective_epochs.csv', np.asarray(self.L))
        
    def build_omega(self, ratings):
        # list of tuples - indices of observed elements in sparse matrix
        print('building omega | {}'.format(datetime.datetime.now()))
        omega = []
        for i in range(ratings.shape[0]):
            omega.append((int(ratings[i,0]), int(ratings[i,1])))
        return omega
    
    def build_omega_u(self, ratings): 
        # omega_u is a list of lists - the ith element is a list containing the indices of
        # the objects that the ith user has rated
        print('building omega_u | {}'.format(datetime.datetime.now()))
        omega_u = []
        for i in range(self.N1):
            user_ratings = ratings[ratings[:,0] == i]
            omega_u.append(list(np.unique(user_ratings[:,1]).astype(int)))
        return omega_u
        
    def build_omega_v(self, ratings):     
        # omega_v is a list of lists - the jth element is a list containing the indices of
        # the users that have rated the jth object
        print('building omega_v | {}'.format(datetime.datetime.now()))
        omega_v = []
        for j in range(self.N2):
            user_ratings = ratings[ratings[:,1] == j]
            omega_v.append(list(np.unique(user_ratings[:,0]).astype(int)))
        return omega_v
        
    def build_M(self, ratings):
        # build ratings matrix M - it will be represented by a dictionary where the keys are tuples of the form
        # (user_index, object_index) and the values are the ratings
        print('building the matrix dictionary M | {}'.format(datetime.datetime.now()))
        M = {}
        for i in range(ratings.shape[0]):
            M[(int(ratings[i,0]), int(ratings[i,1]))] = ratings[i,2]
        return M
    
    def execute_training_epochs(self, ratings):
        # initialize vj's randomly
        # u is (N1 x 5) numpy array, v is (N2 x 5) numpy array
        v = np.random.randn(self.N2, self.d)
        u = np.zeros((self.N1, self.d))

        print('beginning training epochs | {}'.format(datetime.datetime.now()))
        L = []
        for iteration in range(self.num_iter):
            if ((iteration+1) % 10 == 0) and self.verbose:
                print('iteration {} of {} | {}'.format(iteration+1, self.num_iter, datetime.datetime.now()))
            # perform updates to u and v
            for i in range(self.N1):
                # update user location
                sum1 = np.zeros((self.d,self.d))
                sum2 = np.zeros((self.d))
                for j in self.omega_u[i]:
                    sum1 += np.outer(v[j,:], v[j,:])
                    sum2 += self.M[(i,j)] * v[j,:]
                u[i,:] = np.linalg.inv(self.lamd*self.sig2*np.identity(self.d) + sum1).dot(sum2)        
            for j in range(self.N2):
                # update object location
                sum1 = np.zeros((self.d,self.d))
                sum2 = np.zeros((self.d))
                for i in self.omega_v[j]:
                    sum1 += np.outer(u[i,:], u[i,:])
                    sum2 += self.M[(i,j)] * u[i,:]
                v[j,:] = np.linalg.inv(self.lamd*self.sig2*np.identity(self.d) + sum1).dot(sum2)

            # calc objective function value and store it
            sum1 = 0
            for pair in self.omega:
                i, j = pair
                sum1 += 1/(2*self.sig2) * (self.M[(i,j)] - u[i,:].dot(v[j,:]))**2
            sum2 = 0
            for i in range(self.N1):
                sum2 += self.lamd/2 * np.linalg.norm(u[i,:])**2
            sum3 = 0
            for j in range(self.N2):
                sum3 += self.lamd/2 * np.linalg.norm(v[j,:])**2
            L.append(-sum1 - sum2 - sum3)
            
        return u, v, L

    def predict(self, request):
        """
        Use computed U and V matrices to make rating predictions
        :param request: pandas dataframe with columns 'userId' and 'movieId' 
        :return: pandas dataframe with 'userId', 'movieId', and 'prediction'
        """
        predictions = []
        for inx, row in request.iterrows():
            userid = row['userId']
            movieid = row['movieId']
            user_exists = userid in self.user_mapping[:, 0]
            movie_exists = movieid in self.obj_mapping[:, 0]
            if (not user_exists) or (not movie_exists):
                if self.verbose:
                    print('user {} exists: {} | movie {} exists: {}'.format(userid, user_exists, movieid, movie_exists))
                predictions.append(None)
                continue
            user_inx = np.where(self.user_mapping[:, 0] == userid)[0][0]
            map_userid = self.user_mapping[user_inx, 1]
            movie_inx = np.where(self.obj_mapping[:, 0] == movieid)[0][0]
            map_movieid = self.obj_mapping[movie_inx, 1]
            prediction = self.u[map_userid, :].dot(self.v[map_movieid, :])
            predictions.append(prediction)
        results = request.copy()
        results['prediction'] = predictions
        return results
