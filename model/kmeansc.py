import numpy as np

def equal_size_kmeans(X, k):
    points = X.shape[0]
    size = points / k
    # Step 1: Initialize the centroids
    centroids = np.random.rand(k, X.shape[1])
    cluster_assignments = np.full(X.shape[0], -1) 
    cluster_sizes = np.zeros(k, dtype=int)

    for _ in range(1):
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
        
        for i in range(k):
            centroids[i] = np.mean(X[cluster_assignments == i].numpy(), axis = 0, keepdims= True)
    
    return cluster_assignments