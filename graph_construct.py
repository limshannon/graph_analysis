import sqlite3
import networkx as nx
import matplotlib.pyplot as plt

# Connect to the database and create a cursor
conn = sqlite3.connect('sports_club.db')
c = conn.cursor()

# Create the nodes table
c.execute('''CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT)''')

# Insert 7 nodes into the table
c.execute("INSERT INTO nodes VALUES (1, 'Player 1')")
c.execute("INSERT INTO nodes VALUES (2, 'Player 2')")
c.execute("INSERT INTO nodes VALUES (3, 'Player 3')")
c.execute("INSERT INTO nodes VALUES (4, 'Player 4')")
c.execute("INSERT INTO nodes VALUES (5, 'Player 5')")
c.execute("INSERT INTO nodes VALUES (6, 'Player 6')")
c.execute("INSERT INTO nodes VALUES (7, 'Player 7')")

# Create the edges table
c.execute('''CREATE TABLE edges (id INTEGER PRIMARY KEY, node1 INTEGER, node2 INTEGER)''')

# Insert edges between the nodes
c.execute("INSERT INTO edges VALUES (1, 1, 4)")
c.execute("INSERT INTO edges VALUES (2, 1, 3)")
c.execute("INSERT INTO edges VALUES (3, 3, 4)")
c.execute("INSERT INTO edges VALUES (4, 2, 5)")
c.execute("INSERT INTO edges VALUES (5, 5, 6)")
c.execute("INSERT INTO edges VALUES (6, 6, 7)")
c.execute("INSERT INTO edges VALUES (7, 1, 5)")
c.execute("INSERT INTO edges VALUES (8, 2, 6)")
c.execute("INSERT INTO edges VALUES (9, 3, 7)")

# Commit the changes to the database
conn.commit()

# Create a directed graph using networkx
G = nx.DiGraph()

# Retrieve the nodes from the database and add them to the graph
c.execute("SELECT * FROM nodes")
for row in c:
    G.add_node(row[0], label=row[1])

# Retrieve the edges from the database and add them to the graph
c.execute("SELECT * FROM edges")
for row in c:
    G.add_edge(row[1], row[2])

# Close the connection to the database
conn.close()

# Draw the graph using matplotlib
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
