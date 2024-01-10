from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(Nations().training.triples)
df.columns = ['h', 'r', 't']

# Generate triples from the graph
tf = TriplesFactory.from_labeled_triples(df.values)

training, testing = tf.split([0.8, 0.2], random_state=42)
result = pipeline(
        training=training,
        testing=testing,
        model = "NTN",
        model_kwargs=dict(embedding_dim=16),
        optimizer = "adam",
        training_kwargs=dict(num_epochs=10, use_tqdm_batch=False),
        random_seed=42,
        device='cpu',
        negative_sampler = 'bernoulli',
        negative_sampler_kwargs = dict(num_negs_per_pos = 3))

losses = result.losses
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
