import networkx as nx

# Create a knowledge graph
G = nx.Graph()

# Add nodes representing different products
G.add_nodes_from(["Product A", "Product B", "Product C",
                  "Product D", "Product E", "Product F", 
                  "Product G", "Product H", "Product I", 
                  "Product J"])

# edges are products that are often purchased together
G.add_edge("Product A", "Product B")
G.add_edge("Product A", "Product C")
G.add_edge("Product B", "Product D")
G.add_edge("Product C", "Product E")
G.add_edge("Product D", "Product F")
G.add_edge("Product E", "Product G")
G.add_edge("Product F", "Product H")
G.add_edge("Product G", "Product I")
G.add_edge("Product H", "Product J")

# Function to recommend products
def recommend_products(product):
    # Find the neighbors of the input product
    neighbors = list(nx.neighbors(G, product))
    # Return the list of recommended products
    return neighbors

# Example usage
print(recommend_products("Product A")) 
# Returns: ['Product B', 'Product C']
