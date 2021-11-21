# %%
import cv2
import igraph as ig 
import logging as log
from spmodule.splib import *
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
elif True: 
  width = 100
  height = 50
  step = 10

  start = wh2vid(5,5, width)
  end =  wh2vid(95,45, width)

  graph = ig.load("graphs/optimization.graphml")
  
  with open("graphs/optimization.npy", 'rb') as f:
    img = np.load(f)
else:
  width = 100
  height = 50
  step = 10

  start = wh2vid(5,5, width)
  end =  wh2vid(95,45, width)

  no_mountains = 6
  mountain_height = 10
  wall_percent = 15
  graph, img = generate_weighted_graph(width, height, step, start, end, no_mountains, mountain_height, wall_percent)
  
  ig.save(graph, "graphs/optimization.graphml")
  with open("graphs/optimization.npy", 'wb') as f:
    np.save(f, img)

if True:
  LOG.setLevel(log.DEBUG)
  target = end
  
  nnodes = width*height
  lprint("Test for " + str(nnodes) + " nodes")
  
  timer = Timer()
  
  path_ig = graph.get_shortest_paths(start,target)[0]
  time_ig = str(timer.time())
  lprint("Igraph default finished")
  
  path_grd = bestfirst(graph,start,target)
  time_grd = str(timer.time())
  lprint("Best first greedy algorithm finished")
  
  # path_bf = bellfo(graph,start,target)
  # time_bf = str(timer.time())
  # lprint("Bellman-Ford finished")
  
  path_dj = dijkstra(graph,start,target)
  time_dj = str(timer.time())
  lprint("Dijkstra finished")
  
  path_as = Astar(graph,start,target)
  time_as = str(timer.time())
  lprint("A* finished")
  
  path_ant = antss(graph,start,target,3,20)
  time_ant = str(timer.time())
  lprint("Ants finished")
  
         
  cost_ig = path_cost(graph, path_ig)
  lprint("Igraph default: " + time_ig + "s, path cost: " + str(cost_ig))

  cost_grd = path_cost(graph, path_grd)
  lprint("Best first greedy algorithm: " + time_grd + "s, path cost: " + str(cost_grd))
  
  # cost_bf = path_cost(graph, path_bf)
  # lprint("Bellman-Ford: " + time_bf + "s, path cost: " + str(cost_bf))

  cost_dj = path_cost(graph, path_dj)
  lprint("Dijkstra: " + time_dj + "s, path cost: " + str(cost_dj))

  cost_as = path_cost(graph, path_as)
  lprint("A*: " + time_as + "s, path cost: " + str(cost_as))

  cost_ant = path_cost(graph, path_ant)
  lprint("Ants: " + time_ant + "s, path cost: " + str(cost_ant))
  
  for path in [path_ig, path_grd, path_dj, path_as, path_ant]:
    for v in path:
      update_frame(width, step, v ,img , 'w')
      
  cv2.imshow('frame', np.uint8(img))

  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()

  # LOG.setLevel(log.ERROR)
  # print (antss(graph,0,108,30))
  # t5 = time()
  # print("Ants with ERROR log: " + str(t5 - t4))
# %%

if False:
  target = end

  path_ant = antss(graph,start,target,7,1)
  for v in path_ant:
    update_frame(width, step, v ,img , 'w')
  cv2.imshow('frame', np.uint8(img))

  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
    
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