import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

'''
GRAPH CONSTRUCTION
'''
# synthetic graph
G = nx.connected_watts_strogatz_graph(500, 4, 0.5, seed=42)
random.seed(2023)
# Add node attributes to the nodes
for node in G.nodes():
    G.nodes[node]['age'] = np.random.randint(18, 65)
    G.nodes[node]['sex'] = np.random.choice(['male', 'female'])
    G.nodes[node]['location'] = np.random.choice(['urban', 'rural'])
    G.nodes[node]['occupation'] = np.random.choice(['student', 'employee', 'self-employed', 'unemployed'])
    G.nodes[node]['tested'] = np.random.choice([0, 1], p=[0.8, 0.2])
    G.nodes[node]['symptoms'] = np.random.choice([0, 1], p=[0.5, 0.5])
    G.nodes[node]['vaccinated'] = np.random.choice([0, 1], p=[0.6, 0.4])
    G.nodes[node]['mobility'] = random.randint(1,10)

# Add contact data
for (u,v) in G.edges():
    G[u][v]['contact'] = random.randint(1,10)

# add labels using some logic
for node in G.nodes():
    mobility = G.nodes[node]['mobility']
    tested = G.nodes[node]['tested']
    vaccinated = G.nodes[node]['vaccinated']
    symptoms = G.nodes[node]['symptoms']

    if mobility > 7 and (tested == 1 or (vaccinated == 1 and random.random() > 0.5) or symptoms == 1 and random.random() < 0.5):
        G.nodes[node]['label'] = 'infected'
    else:
        G.nodes[node]['label'] = 'not infected'


'''
CUSTOM DATA HANDLER
'''
# create edge index of the graph
edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()

# create the edge weight tensor
edge_weights = [G[u][v]['contact'] for u, v in list(G.edges())]
edge_weight = torch.tensor(edge_weights, dtype=torch.float)

# create node features 
# Create a dataframe from the graph nodes
df = pd.DataFrame(dict(G.nodes(data=True))).T

# convert selected features to tensor
node_features = torch.tensor(df[['tested','symptoms',
                                 'vaccinated','mobility']].astype(float).values,
                             dtype=torch.float)

# labels
node_labels = df.label.map({'infected': 1, 'not infected': 0})
y = torch.from_numpy(node_labels.values).type(torch.long)

# create train and test masks
X_train, X_test, y_train, y_test = train_test_split(pd.Series(G.nodes()), 
                                                    node_labels,
                                                    stratify = node_labels,
                                                    test_size=0.20, 
                                                    random_state=56)

n_nodes = G.number_of_nodes()
train_mask = torch.zeros(n_nodes, dtype=torch.bool)
test_mask = torch.zeros(n_nodes, dtype=torch.bool)
train_mask[X_train.index] = True
test_mask[X_test.index] = True

# create torch_geometric Data object
data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight,
            y=y, train_mask=train_mask, test_mask=test_mask,
            num_classes = 2, num_features=len(node_features))


'''
GRAPH NEURAL NETWORK TRAINING AND EVALUATION
'''

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = ChebNet(data.num_features, 32, data.num_classes, K=2)
model, data = model.to(device), data.to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

# test function 
@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

# train for 100 epochs and evaluate
for epoch in range(1, 101):
    loss = train()
    train_acc, test_acc = test()

print("*"*10)
print("Training accuracy: {:.2f}%".format(train_acc*100))
print("Testing accuracy: {:.2f}%".format(test_acc*100))
print("*"*10)
