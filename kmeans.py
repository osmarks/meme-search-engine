import numpy as np
import msgpack
import math
import torch
from torch import autograd
from torch.nn.modules import distance

n_dims = 1152
data = np.fromfile("500k_vecs.bin", dtype=np.float16).reshape(-1, n_dims).astype(np.float32)
n_clusters = 42

def partition_soft(vectors, k, max_iter=100, batch_size=8192):
    n_vectors = len(vectors)
    perm = torch.randperm(n_vectors)
    centroids = vectors[perm[:k]]
    biases = torch.randn(k, device=vectors.device)
    centroids.requires_grad = True
    biases.requires_grad = True
    opt = torch.optim.Adam([centroids], lr=0.01)
    temperature = 1.0
    size_scale = 15.0
    bias_scale = 1.0
    score_scale = 0.1

    desired_size = n_vectors / k

    with autograd.detect_anomaly():
        for it in range(max_iter):
            cluster_sizes = torch.zeros(k, device=vectors.device)
            norm_centroids = torch.nn.functional.normalize(centroids)
            score = torch.zeros(k, device=vectors.device)
            #soft_assignment_entropy = 0

            for i in range(0, n_vectors, batch_size):
                batch = vectors[i:i+batch_size]

                similarities = torch.matmul(batch, norm_centroids.T)

                soft_assignments = ((similarities + biases) / temperature).softmax(dim=1)
                #print(soft_assignments[0])

                #print(soft_assignments.shape, similarities.shape)

                #entropy_by_vector = -soft_assignments.mul(soft_assignments.log2()).sum(dim=1)
                #soft_assignment_entropy += entropy_by_vector.mean()
                cluster_sizes += soft_assignments.sum(dim=0)
                score += similarities.mean(dim=0)

            opt.zero_grad()

            distances_from_ideal_cluster_size = (cluster_sizes - desired_size).pow(2) / (desired_size ** 2)
            size_loss = distances_from_ideal_cluster_size.mean()
            bias_loss = biases.pow(2).mean()
            score_loss = -score.mean()
            loss = size_scale * size_loss + bias_scale * bias_loss + score_scale * score_loss
            loss.backward()
            opt.step()

            print(temperature, size_scale * size_loss.detach().tolist(), bias_scale * bias_loss.detach().tolist(), score_scale * score_loss.detach().tolist(), cluster_sizes.tolist())

            #centroids = new_centroids / cluster_sizes.unsqueeze(1)

            #if torch.allclose(centroids, new_centroids, rtol=1e-4):
            #    break

            if it % 100 == 0:
                temperature *= 0.999
                size_scale *= 1.1

        return centroids.detach().cpu().numpy(), biases.detach().cpu().numpy()

SPILL_K = 2
def simulated_annealing(vectors, k, max_iter=100, batch_size=31768):
    n_vectors = len(vectors)
    centroids = torch.randn(k, n_dims, device=vectors.device)
    desired_size = n_vectors / k

    def fitness(centroids):
        cluster_sizes = torch.zeros(SPILL_K, k, device=vectors.device, dtype=torch.int32)
        norm_centroids = torch.nn.functional.normalize(centroids)

        for i in range(0, n_vectors, batch_size):
            batch = vectors[i:i+batch_size]

            similarities = torch.matmul(batch, norm_centroids.T)
            values, indices = similarities.topk(SPILL_K, dim=1)

            for j in range(SPILL_K):
                batch_counts = torch.bincount(indices[:, j], minlength=k)
                cluster_sizes[j] += batch_counts

        distances_from_ideal_cluster_size = (cluster_sizes - desired_size).abs()
        #print(distances_from_ideal_cluster_size)
        return distances_from_ideal_cluster_size.max(), distances_from_ideal_cluster_size.argmax(dim=1)

    global_best, global_best_result = None, 1000000

    temperature = 1.0

    last_fitness, _ = fitness(centroids)
    last_improvement = 0
    for _ in range(max_iter):
        n = centroids + torch.randn_like(centroids) * temperature
        new_fitness, worst_centroid = fitness(n)
        print(last_fitness.tolist(), new_fitness.tolist(), temperature)
        if new_fitness < last_fitness:
            centroids = n
            temperature *= 0.999
            last_fitness = new_fitness
            last_improvement = 0
        else:
            temperature *= 0.9995
            last_improvement += 1
        if last_improvement > 100:
            print("rerolling")
            centroids[worst_centroid] = torch.randn_like(centroids[worst_centroid])
            last_improvement = 0
            temperature *= 1.1
            last_fitness = new_fitness
        if last_fitness < desired_size * 0.1:
            break
        temperature = min(1.5, temperature)
        if new_fitness < global_best_result:
            global_best = n
            global_best_result = new_fitness

    return torch.nn.functional.normalize(centroids)

""""
centroids = list(zip(*partition(torch.tensor(data), n_clusters**2, max_iter=100)))

BALANCE_WEIGHT = 3e-2

big_clusters = [ ([x], c) for x, c in centroids[:n_clusters] ]
centroids = centroids[n_clusters:]

while centroids:
    avg_size = sum(c for _, c in big_clusters) / len(big_clusters)

    for i, (items, count) in enumerate(big_clusters):
        def match_score(x):
            return 1/len(items) * sum(np.dot(x, y) for y in items)

        candidate_index, candidate = max(enumerate(centroids), key=lambda x: match_score(x[1][0]) - BALANCE_WEIGHT * max(0, count + x[1][1] - avg_size))
        centroids.pop(candidate_index)
        big_clusters[i] = (items + [candidate[0]], count + candidate[1])

print([x[1] for x in big_clusters])

"""

centroids = simulated_annealing(torch.tensor(data, device=torch.device("cuda")), n_clusters, max_iter=80000).detach().cpu().numpy()
centroids.astype(np.float16).tofile("centroids.bin")
