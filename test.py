# %%
import igraph as ig 
import logging as log
from splib import *
from time import time
from random import randint

# remember to set level to WARNING before actual tests!
log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')
LOG = log.getLogger()

if False:
  graph_name = "graphs/basic.graphml"
  # graph_name = "graphs/p2p-Gnutella08.graphml"

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
else: 
  width = 40
  height = 40

  start = wh2vid(1,1, width)
  end =  wh2vid(38,39, width)

  graph = generate_graph(width, height)

  for v in graph.vs():
    v["distance"] = np.linalg.norm(np.array(vid2wh(v.index, width)) - np.array(vid2wh(end, width)))
  graph.vs[end]["distance"] = 0.1

  for e in graph.es():
      e["weight"] = graph.vs[e.target]["distance"]
  ig.save(graph, "graphs/basic.graphml")

if True:
  target = end
  
  timer = Timer()
  print(graph.get_shortest_paths(0,target))
  print("Igraph default: " + str(timer.time()))

  print (bellfo(graph,0,target))
  print("Bellman-Ford: " + str(timer.time()))

  print (dijkstra(graph,0,target))
  print("Dijkstra: " + str(timer.time()))

  print (Astar(graph,0,target))
  print("A*: " + str(timer.time()))

  print (antss(graph,0,target,30))
  print("Ants: " + str(timer.time()))

  # LOG.setLevel(log.ERROR)
  # print (antss(graph,0,108,30))
  # t5 = time()
  # print("Ants with ERROR log: " + str(t5 - t4))
# %%
if False:
  LOG.setLevel(log.ERROR)
  # Accuracy tester for Ants #
  # ~90% for first try at 10 generations with small graph
  # ~60% first try with bigger graph - 20 nodes\
  # back to ~90% for bigger graph after some adjustements 

  # paths = graph.get_all_shortest_paths(0,18)
  paths = [[0, 3, 16, 12, 15, 18]]

  print(paths)

  for g_num in range(30,45,5):
    hits = 0
    for i in range(100):
      if antss(graph,0,18,g_num) in paths:
        hits += 1
    print("Accuracy for " + str(g_num) + " generations: " + str(hits/100))