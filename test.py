# %%

from os import path
import igraph as ig 
import logging as log
from splib import *
from time import time

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')

graph_name = "graphs/basic.graphml"

try:
  graph = ig.load(graph_name)
  log.info("Succesfully loaded graph")
except FileNotFoundError:
  log.exception("Could not find file to load graph")

  graph = ig.Graph()
  graph.add_vertices(10)
  graph.add_edges([(0,1), (1,2), (1,7), (2,3),
               (2,4), (2,5), (4,9), (5,6), 
               (5,7), (5,8), (6,8), (8,9)])
  
  layout = graph.layout_kamada_kawai()
  ig.plot(graph, layout=layout)

  ig.save(graph, "graphs/basic.graphml")
except :
  log.exception("Could not load graph from file")

if False:
  t_start = time()
  print(graph.get_shortest_paths(0,5))
  t1 = time()
  print(t1 - t_start)

  print (dijkstra(graph,0,5))
  t2 = time()
  print(t2 - t1)

  print (bellfo(graph,0,5))
  t3 = time()
  print(t3 - t2)

  print (antss(graph,0,5))
  t4 = time()
  print(t4 - t3)
# %%
if False:
  # Accuracy at ~ 90% for first try
  paths = graph.get_all_shortest_paths(0,5)
  print(paths)

  hits = 0
  for i in range(1000):
    if antss(graph,0,5) in paths:
      hits += 1
  print("Accuracy: " + str(hits/1000))