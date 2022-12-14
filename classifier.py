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
        self.N, self.V = xdata.shape
        # N-len dict - mapping the positions in the matrix to the book's title
        self.books = books
        # number of classes
        self.K, self.D = K, D
        self.centers = np.random.randint(0,self.V, size=(K,D))
    
    # Assume the dimensions represent the top 50 words
    # each point represents a document
    # their values for each dim range from 0-V exclusive
    def KMeans(self, iterr=0):
            clusters = [[] for i in range(self.K)]
    newcenters = np.zeros(self.centers.shape)

    print(iterr)
#     print('center.shape'+str(self.centers.shape))
#     print(self.centers)
    #assume centers are KxD and xdata is NxD
    t = np.concatenate(tuple([self.xdata]*self.K), axis=1).reshape((self.N*self.K,self.D))
    c = np.concatenate(tuple([self.centers]*self.N), axis=0).reshape((self.N*self.K,self.D))
    dist = np.sum((t-c)**2,axis=1)
    dist = dist.reshape(self.N,self.K)
    
    for doc,cluster in enumerate(np.argmin(dist,axis=1)):
        clusters[cluster].append(self.xdata[doc])
    
    for n,cluster in enumerate(clusters):
        if len(cluster) == 0:
            newcenters[n]=self.centers[n]
        else:
            newcenters[n]=np.mean(np.asarray(cluster),axis=0)
    if (self.centers == newcenters).all():
        return clusters
    self.centers = newcenters
    return self.KMeans(iterr+1)

    def classify(self, testing):
        if testing.shape[1]!=self.D:
            print(f"Error: tesing shape{testing.shape} Must have Dimensionality of {self.D} for top M words")
            return None
        distances = []
        t = np.concatenate(tuple([testing]*self.K), axis=1).reshape((testing.shape[0]*self.K,self.D))
        c = np.concatenate(tuple([self.centers]*testing.shape[0]))
        dist = np.sum((t-c)**2,axis=1)
        return np.argmin(dist.reshape(self.K,testing.shape[0],testing.shape[1]),axis=0)