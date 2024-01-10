import networkx as nx

# Create a new graph
G = nx.Graph()

# Add nodes representing products
products = ["Product A", "Product B", "Product C", 
            "Product D", "Product E", "Product F", 
            "Product G", "Product H", "Product I", 
            "Product J"]
G.add_nodes_from(products)

# Add edges and weights between products that are often purchased together
G.add_edge("Product A", "Product B", weight=0.8)
G.add_edge("Product A", "Product C", weight=0.7)
G.add_edge("Product B", "Product D", weight=0.9)
G.add_edge("Product C", "Product E", weight=0.6)
G.add_edge("Product D", "Product F", weight=0.8)
G.add_edge("Product E", "Product G", weight=0.7)
G.add_edge("Product F", "Product H", weight=0.9)
G.add_edge("Product G", "Product I", weight=0.6)
G.add_edge("Product H", "Product J", weight=0.8)

# Define a dictionary to store the popularity of each product
popularity = {"Product A": 100, "Product B": 90, "Product C": 80, 
              "Product D": 70, "Product E": 60, "Product F": 50, 
              "Product G": 40, "Product H": 30, "Product I": 20, 
              "Product J": 10}

# Function to recommend products
def recommend_products(product, n=2):
    # Find the neighbors 
    neighbors = nx.neighbors(G, product)
    # Create a list to store the recommendations
    recommendations = []
    # Iterate through the neighbors
    for neighbor in neighbors:
        # Get the weight of the edge between the input product and the neighbor
        weight = G[product][neighbor]['weight']
        # Get the popularity of the neighbor
        pop = popularity[neighbor]
        # Compute a score based on the weight and popularity
        score = weight * pop
        # Add the neighbor and its score to the list of recommendations
        recommendations.append((neighbor, score))
    # Sort the recommendations by score
    recommendations.sort(key=lambda x: x[1], reverse=True)
    # Return the top n recommendations
    return [x[0] for x in recommendations[:n]]

# Example for product A
print(recommend_products("Product B")) 
