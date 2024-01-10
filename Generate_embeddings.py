import networkx as nx
import numpy as np
from random import shuffle
from sklearn.linear_model import SGDRegressor

# scoring function
def scoring_function(embeddings, edges):
    return np.sum(embeddings[edges[:, 0]] * embeddings[edges[:, 1]], axis=1)

# loss function
def loss_function(scores, labels):
    return ((scores - labels) ** 2).mean()

# optimizer
def optimizer(embeddings, edges, labels, learning_rate):
    scores = scoring_function(embeddings, edges)
    loss = loss_function(scores, labels)
    sgd = SGDRegressor(learning_rate='constant', eta0=learning_rate)
    sgd.partial_fit(embeddings[edges[:, 0]], scores - labels)
    embeddings -= sgd.coef_
    return loss, embeddings

# negative generation function
def negative_generation(edges, num_negatives):
    all_negatives = np.random.randint(0, embeddings.shape[0], size=(num_negatives, 2))
    return np.concatenate([edges, all_negatives], axis=0)

# Create a graph with 5 nodes
graph = nx.Graph()
graph.add_nodes_from(range(5))

# Add some edges to the graph
graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

# number of dimensions for the embeddings
embedding_dim = 128

# Initialize the embeddings randomly
embeddings = np.random.rand(graph.number_of_nodes(), embedding_dim)

# number of negative samples
num_negatives = 10

# learning rate
learning_rate = 0.01

# number of iterations
num_iterations = 100

# Get the edges of the graph
edges = np.array(graph.edges())

# Create the labels for the positive edges
labels = np.ones(edges.shape[0])

# Generate the negative edges
negative_edges = negative_generation(edges, num_negatives)

# Concatenate the positive and negative edges
all_edges = np.concatenate([edges, negative_edges], axis=0)

# Concatenate the labels for the positive and negative edges
all_labels = np.concatenate([labels, np.zeros(num_negatives)], axis=0)

# Shuffle the edges and labels
shuffle_indices = np.random.permutation(np.arange(edges.shape[0]))
all_edges = all_edges[shuffle_indices]
all_labels = all_labels[shuffle_indices]

for i in range(num_iterations):
    loss, embeddings = optimizer(embeddings, all_edges, all_labels, learning_rate)
    # Print the loss every 10 iterations
    if i % 10 == 0:
        print("Iteration: {} Loss: {}".format(i, loss.round(5)))
print("*"*10)
print("Shape of emebddings: ",embeddings.shape)
