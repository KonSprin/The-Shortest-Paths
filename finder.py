# %%
import igraph as ig 
import logging as log
import time

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')


graph_name = "graphs/p2p-Gnutella08.graphml"

# %% Loading graph
try:
  graph = ig.load(graph_name)
  log.info("Succesfully loaded graph")
except FileNotFoundError:
  log.exception("Could not find file to load graph")
except :
  log.exception("Could not load graph from file")

# %%

start = time.time()
x = []
for v in range(graph.vcount()):
    oneloop = time.time()
    x.append(graph.get_shortest_paths(v))
    log.debug("Loop nr: " + str(v) + " \nTime: " + str(oneloop - time.time()))
print(start - time.time())

