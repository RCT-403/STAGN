import numpy as np

def equal_size_kmeans(X, k, runs=3, spectral=0):
    points = X.shape[0]
    size = points / k

    centroids = np.random.rand(k, X.shape[1])

    for _ in range(runs):
        cluster_assignments = np.full(X.shape[0], -1) 
        cluster_sizes = np.zeros(k, dtype=int)

        for i in range(points):
            cluster_assignments[i] = np.argmax(np.matmul(X[i], centroids.T))
            cluster_sizes[cluster_assignments[i]] += 1
        
        for j in range(k):
            while cluster_sizes[j] > size:
                excess_indices = np.where(cluster_assignments == j)[0]
                if len(excess_indices) == 0:
                    break
                excess_index = excess_indices[-1] 
                cluster_sizes[j] -= 1    

                for m in range(k):
                    if cluster_sizes[m] < size:
                        cluster_assignments[excess_index] = m
                        cluster_sizes[m] += 1
                        break
                        
        if not spectral:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i].numpy(), axis = 0, keepdims= True)

        else:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i], axis = 0, keepdims= True)

    return cluster_assignments