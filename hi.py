from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = StandardScaler().fit_transform(data.data)

model = DBSCAN(eps=0.8, min_samples=6)
labels = model.fit_predict(X)

print(labels)

