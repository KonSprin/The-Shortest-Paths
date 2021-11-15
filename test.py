# %%
import igraph as ig 
import logging as log
from splib import *
from json import dumps as jdump
from sys import stdout

# remember to set level to WARNING before actual tests!
logging_type = "file_only"

if logging_type == "file_only":
  log.basicConfig(level=log.DEBUG,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')
elif logging_type == "stdout+logging":
  file_handler = log.FileHandler(filename='sp.log')
  stdout_handler = log.StreamHandler(stdout)
  handlers = [file_handler, stdout_handler]

  log.basicConfig(level=log.DEBUG,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  handlers=handlers)

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
  width = 100
  height = 100

  start = wh2vid(1,50, width)
  end =  wh2vid(999,40, width)

  graph = generate_graph(width, height)

  deleted_vs = []
  for w,h in draw_line(50,4,50,85):
    v = wh2vid(w,h,width)
    graph.delete_edges(graph.incident(v))
    deleted_vs.append(v)
  graph["deleted_vs"] = jdump(deleted_vs)
    
  for v in graph.vs():
    v["distance"] = np.linalg.norm(np.array(vid2wh(v.index, width)) - np.array(vid2wh(end, width)))
  graph.vs[end]["distance"] = 0.1

  for e in graph.es():
      # e["weight"] = graph.vs[e.target]["distance"]
      e["weight"] = 1
  ig.save(graph, "graphs/basic.graphml")

if True:
  target = end
  
  nnodes = width*height
  lprint("Test for " + str(nnodes) + " nodes")
  
  timer = Timer()
  path_length = str(len(graph.get_shortest_paths(start,target)[0]))
  lprint("Igraph default: " + str(timer.time()) + "s, path length: " + path_length)

  path_length = str(len(bestfirst(graph,start,target)))
  lprint("Best first greedy algorithm: " + str(timer.time()) + "s, path length: " + path_length)
  
  path_length = str(len(bellfo(graph,start,target)))
  lprint("Bellman-Ford: " + str(timer.time()) + "s, path length: " + path_length)

  # path_length = str(len(dijkstra(graph,start,target)))
  # print("Dijkstra: " + str(timer.time()) + "s, path length: " + path_length)

  path_length = str(len(Astar(graph,start,target)))
  lprint("A*: " + str(timer.time()) + "s, path length: " + path_length)

  path_length = str(len(antss(graph,start,target,30)))
  lprint("Ants: " + str(timer.time()) + "s, path length: " + path_length)

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