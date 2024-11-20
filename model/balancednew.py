import numpy as np

def equal_size_kmeans(X, k, runs=3, spectral=0):
    points = X.shape[0]
    size = points / k

    centroids = np.random.rand(k, X.shape[1])

    for _ in range(runs):
        cluster_assignments = np.full(X.shape[0], -1) 
        cluster_sizes = np.zeros(k, dtype=int)
        pairwise_distance = np.matmul(X, centroids.T)
        for i in range(points):
            cluster_assignments[i] = np.argmax(pairwise_distance[i])
            cluster_sizes[cluster_assignments[i]] += 1
        
        excess_indices = np.where(cluster_sizes > size)[0]
        less_indices = np.where(cluster_sizes < size)[0]
        while len(excess_indices) > 0:
            for i in excess_indices:
                all_indices = np.where(cluster_assignments == i)[0]
                distances = pairwise_distance[all_indices][i]
                sorted_indices = np.argsort(distances)
                num_to_move = int(cluster_sizes[i] - size)
                indices_to_move = sorted_indices[-num_to_move:]
                for j in indices_to_move:
                    insert = np.argmax(pairwise_distance[j][less_indices])
                    cluster_assignments[j] = insert
                    cluster_sizes[i] -= 1
                    cluster_sizes[insert] += 1

            excess_indices = np.where(cluster_sizes > size)[0]

        # for j in range(k):
        #     while cluster_sizes[j] > size:
        #         excess_indices = np.where(cluster_assignments == j)[0]
        #         if len(excess_indices) == 0:
        #             break
        #         excess_index = excess_indices[-1] 
        #         cluster_sizes[j] -= 1    

        #         for m in range(k):
        #             if cluster_sizes[m] < size:
        #                 cluster_assignments[excess_index] = m
        #                 cluster_sizes[m] += 1
        #                 break
                        
        if not spectral:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i].numpy(), axis = 0, keepdims= True)

        else:
            for i in range(k):
                centroids[i] = np.mean(X[cluster_assignments == i], axis = 0, keepdims= True)

    return cluster_assignments