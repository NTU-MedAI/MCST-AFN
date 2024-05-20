import torch
import torch.nn as nn
from torch import sin, cos
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rot_z(gamma):
    return torch.tensor([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


class Classifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(),
                                 nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        return self.out(torch.cat((x1, x2), dim=-1)).squeeze(-1)


class ConformationSorter:
    def __init__(self, num_frames=10000, num_embeddings=6, embedding_dim=128):
        self.num_frames = num_frames
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.sorted_indices = None

    def fit(self, frames, embeddings):
        assert frames.shape[0] == embeddings.shape[0] == self.num_embeddings
        assert embeddings.shape[1] == self.embedding_dim
        self.embeddings = embeddings
        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, -1)  # ignore self-similarity
        self.sorted_indices = np.argsort(similarities.sum(axis=1))[::-1]

    def predict(self):
        if self.embeddings is None or self.sorted_indices is None:
            raise ValueError("Model not trained yet")
        return self.sorted_indices

    def calculate_loss(self, frames, embeddings):
        assert frames.shape[0] == embeddings.shape[0] == self.num_embeddings
        assert embeddings.shape[1] == self.embedding_dim
        # Calculate cosine similarity between embeddings
        similarities = cosine_similarity(embeddings)
        # Ignore self-similarity
        np.fill_diagonal(similarities, -1)
        # Calculate the sum of similarities for each embedding
        sums = similarities.sum(axis=1)
        # Calculate the sorted indices based on the sums
        sorted_indices = np.argsort(sums)[::-1]
        # Calculate the mean distance between consecutive embeddings in the sorted order
        distances = np.linalg.norm(embeddings[sorted_indices[1:]] - embeddings[sorted_indices[:-1]], axis=1)
        mean_distance = np.mean(distances)
        return mean_distance






