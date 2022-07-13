
import numpy as np
import sklearn.datasets as data
from scipy import stats

from sklearn import neighbors
class knn:

    def __init__(self, n_neighbors):
        if n_neighbors < 1:
            raise ValueError("n_neighbors can't be less than 1")
        self.n_neighbors = n_neighbors # need error checking  > n_samples

    def fit(self, X, y):
        self.X = np.ma.array(X)
        self.y = y

    def predict(self, X):

        predictions = np.full(self.y.shape, fill_value=-1)

        i = 0
        while i < len(X):
            j = 0
            miniPredict = np.full(self.n_neighbors, fill_value=-1)
            prediction  = None
            mask = np.zeros(self.X.shape)
            while j < self.n_neighbors:
                idx = np.abs(self.X - X[i]).sum(axis=1).argmin()
                mask[idx] = [1, 1, 1, 1]
                self.X.mask = mask
                # if self.n_neighbors == 1:
                #     prediction = self.y[idx]
                #     break
                miniPredict[j] = self.y[idx]
                j+=1


            prediction = stats.mode(miniPredict).mode[0]

            predictions[i] = prediction
            i+=1
        return predictions




if __name__ == "__main__":

    clf = knn(n_neighbors = 5)
    iris = data.load_iris()
    clf.fit(iris.data, iris.target)
    predictions1 = clf.predict(iris.data)

    skKnn = neighbors.KNeighborsClassifier(5, metric='manhattan')
    skKnn.fit(iris.data, iris.target)
    predictions2 = skKnn.predict(iris.data)

    i = 0