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

    def filter_rating_freq(self, freq, threshold=200, movieId_col='movieId'):
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