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
  graph.add_vertices(19)
  graph.add_edges([(0,1), (1,2), (1,7), (2,3),
               (2,4), (2,5), (4,9), (5,6), 
               (5,7), (5,8), (6,8), (8,9),
               (3,10), (10,11), (10,13), (10,17),
               (9,13), (9,17), (3,16), (16,12),
               (12,15), (12,14), (15,18), (14,18),
               (17,12), (0,3), (0,18)])
  graph.es["weight"] = [1, 2, 6, 3, 2, 1,
                         9, 3, 1, 4, 7, 1,
                         11, 4, 5, 4, 2, 5,
                         4, 2, 3, 4, 5, 6,
                         2, 1, 100]
  
  # layout = graph.layout_kamada_kawai()
  # ig.plot(graph, layout=layout)

  ig.save(graph, "graphs/basic.graphml")
except :
  log.exception("Could not load graph from file")

if True:
  t_start = time()
  print(graph.get_all_shortest_paths(0,18))
  t1 = time()
  print("Igraph default: " + str(t1 - t_start))

  print (bellfo(graph,0,18))
  t2 = time()
  print("Bellmond-Ford: " + str(t2 - t1))

  print (dijkstra(graph,0,18))
  t3 = time()
  print("Dijkstra: " + str(t3 - t2))

  print (antss(graph,0,18,30))
  t4 = time()
  print("Ants: " + str(t4 - t3))
# %%
if False:
  # Accuracy tester for Ants #
  # ~90% for first try at 10 generations with small graph
  paths = graph.get_all_shortest_paths(0,18)
  print(paths)

  for g_num in range(95,100):
    hits = 0
    for i in range(100):
      if antss(graph,0,18,g_num) in paths:
        hits += 1
    print("Accuracy for " + str(g_num) + " generations: " + str(hits/100))