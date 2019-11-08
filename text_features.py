from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse
import os


class TextFeaturesGenerator:

    def __init__(self,text_series=None):
        """
        :param text_series: A pandas series with the text
        """
        self.text_series = text_series
        self.count_vectorizer = None
        self.tfidf_vectorizer = None

        self.bow_mat = None
        self.tfidf_mat = None

    def get_bow_matrix(self):
        """
        Returns:
            bow_matrix: A CSR (Compressed Sparse Row Matrix) of bag-of-words representation
            of the matrix
        """
        if self.bow_mat is None:
            self.count_vectorizer = CountVectorizer()
            self.bow_mat = self.count_vectorizer.fit_transform(self.text_series)
        return self.bow_mat

    def get_tfidf_matrix(self):
        """
        Returns:
            bow_matrix: A CSR (Compressed Sparse Row Matrix) of tf-idf representation
            of the matrix
        """
        if self.tfidf_mat is None:
            if self.bow_mat is None:
                _ = self.get_bow_matrix()
            self.count_vectorizer = CountVectorizer()
            self.tfidf_vectorizer = TfidfTransformer(use_idf=True).fit(self.bow_mat)
            self.tfidf_mat = self.tfidf_vectorizer.transform(self.bow_mat)
        return self.tfidf_mat

    def save_matrices(self,folder=""):
        """
        Arguments:
        :param folder: Folder / directory in which to save the matrices
                        Will save in current working folder if not specified
        """
        if self.bow_mat is None:
            _ = self.get_bow_matrix()
        if self.tfidf_mat is None:
            _ = self.get_tfidf_matrix()
        bow_location = os.path.join(folder, "bow_mat.npz")
        tfidf_location = os.path.join(folder,"tfidf_mat.npz")
        sparse.save_npz(bow_location, self.bow_mat)
        sparse.save_npz(tfidf_location,self.tfidf_mat)





