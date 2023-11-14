import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs( n_samples=700, centers=2, cluster_std=2.75, random_state=741)
features[:8]

#print(features)

#print(true_labels)


scaled_features = StandardScaler().fit_transform(features)

#print(scaled_features)

kmeans =KMeans (init="random", n_clusters=2, n_init=10, max_iter=100, random_state=348)

kmeans.fit(scaled_features)

print("kmeans inertia: "+str(kmeans.inertia_)+'\n'+"cluster centers: "+str( kmeans.cluster_centers_))


kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)




plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
