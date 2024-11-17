from balanced_kmeans import kmeans_equal
import torch
N = 10000
batch_size = 10
num_clusters = 100
device = 'cuda'
if not torch.cuda.is_available():
    device = 'cpu'

cluster_size = N // num_clusters
X = torch.rand(batch_size, N, dim, device=device)
choices, centers = kmeans_equal(X, num_clusters=num_clusters)