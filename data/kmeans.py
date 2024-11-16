import torch
from kmeansc import *
import time
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score

def map_arrangement(arr):
    # Initialize the result array with None
    result = [None] * len(arr)

    # Variable to track the next number to assign
    num = 1  # Starting from 1

    # First pass to assign unique numbers
    for i, number in enumerate(arr):
        if result[i] is None:
            result[i] = num  # Assign the current number
            for j in range(i + 1, len(arr)):  # Start from the next index
                if arr[j] == number and result[j] is None:
                    result[j] = num  # Assign the same number to duplicates
            num += 1  # Increment for the next unique number

    return result

address = './data/SE(PeMS)_52.txt'

def check_type(obj):
    if isinstance(obj, np.ndarray):
        print("This is a NumPy array.")
    elif isinstance(obj, list):
        print("This is a Python list.")
    elif isinstance(obj, torch.Tensor):
        print("This is a PyTorch tensor.")
    else:
        print("This is an unknown type.")

with open(address, mode='r') as f:
    lines = f.readlines()
    temp = lines[0].split(' ')
    num_vertex, dims = int(temp[0]), int(temp[1])
    SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

labels = equal_size_kmeans(SE, 65)
check_type(labels)
print(labels)
print(labels.dtype)

'''
start_time = time.time()

for i in range(5):
    all_labels = []
    for i in range(10):
        labels = equal_size_kmeans(SE, 65)
        new_labels = map_arrangement(labels)
        all_labels.append(new_labels)

    total_nmi = 0
    total_ari = 0
    for i in range(10):
        for j in range(10):
            if(i != j):
                total_nmi += normalized_mutual_info_score(all_labels[i], all_labels[j])
                total_ari += adjusted_rand_score(all_labels[i], all_labels[j])
                

    avg_nmi = total_nmi/90
    avg_ari = total_ari/90
    print(avg_nmi)
    print(avg_ari)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

'''
