# links for later

# https://towardsdatascience.com/efficient-k-means-clustering-algorithm-with-optimum-iteration-and-execution-time-9358a794406c#603e
# https://github.com/ahegel/genre-ML-clustering/blob/master/genre_clustering.py
# https://www.youtube.com/watch?v=AQ4zCWn7y-0

import numpy as np

class GenreClassifier:
    # intializes the object
    def __init__(self, xdata, ydata, vocab, books, K, D):
        # a VxN numpy matrix - the tfidf score for each word in each book
        self.xdata = xdata
        # N-len list of strings - ground truth labels for each book
        self.ydata = ydata
        # V-len dict - mapping the positions in the matrix to every unique word
        self.vocab = vocab
        self.V, self.N = xdata.shape
        # N-len dict - mapping the positions in the matrix to the book's title
        self.books = books
        # number of classes
        self.K, self.D = K, D
        self.centers = [np.random.randint(0,self.V, size=(K,D))]
    
    def KNN(self):
        #
        return
    
    # Assume the dimensions represent the top 50 words
    # each point represents a document
    # their values for each dim range from 0-V exclusive
    def KMeans(self, iter=0):
        clusters = [[] for i in range(self.K)]
        newclusters = []
        newcenters = []

        #assume centers are KxD and xdata is NxD
        t = np.concatenate(tuple([self.xdata]*self.K), axis=1).reshape((self.N*self.K,self.D))
        c = np.concatenate(tuple([self.centers]*self.N))
        dist = np.sum((t-c)**2,axis=1)

        dist.reshape(self.N,self.K)

        for doc,cluster in enumerate(np.argmin(dist,axis=1)):
            clusters[cluster].append(self.xdata[doc,:])
        clusters = np.array(clusters)
        newcenters = np.mean(np.array(clusters),axis=0)

        if (self.centers == newcenters).all():
            return self.centers
        self.centers = newcenters
        return self.KMeans(iter+1)

    # creates a new Nx(A+B+1) matrix from xdata containing the vocab indices of certain words
    # A is the number of highest tfidf scores
    # B is the number of lowest tfidf scores
    # the median is also added
    def subset(self, A=40, B=10):
        return

    def distance(self):
        #
        return
    
