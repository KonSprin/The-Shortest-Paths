import logging as log
from spmodule.splib import *
from random import sample, randint
import json

log.basicConfig(level=log.DEBUG,
                format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                filename='sp.log', filemode='w')

LOG = log.getLogger()

dist_ig = []
dist_gr = []
dist_bf = []
dist_dj = []
dist_as = []

times_ig = []
times_gr = []
times_bf = []
times_dj = []
times_as = []

flag_ig = True
flag_gr = True
flag_bf = True
flag_dj = True
flag_as = True

width = 2
height = 2

timer = Timer()

for i in range(300):
  timer.time()
  vcount = width * height
  start, target = sample(range(vcount), 2)

  no_mountains = randint(1, int(vcount/4))
  mountain_height = randint(2, 7)
  wall_percent = randint(0, 30)
  
  lprint(f"Iteration {i} for {vcount} nodes")
  graph = generate_weighted_graph_noimg(width, height, start, target, no_mountains, mountain_height, wall_percent)
  
  log.info(f"Graph generation: {timer.time()}s")
  
  if flag_ig:
    path_ig = graph.get_shortest_paths(start,target)[0]
    time_ig = timer.time()
    cost_ig = sum([graph.vs(v)["height"][0] for v in path_ig])
    log.debug("Igraph default: " + str(time_ig) + "s, path cost: " + str(cost_ig))
    dist_ig.append(cost_ig)
    times_ig.append(time_ig)
    timer.time()
    if time_ig > 10:
      flag_ig = False

  if flag_gr:
    path_gr = bestfirst(graph,start,target)
    time_gr = timer.time()
    cost_gr = sum([graph.vs(v)["height"][0] for v in path_gr])
    log.debug("Best first greedy algorithm: " + str(time_gr) + "s, path cost: " + str(cost_gr))
    dist_gr.append(cost_gr)
    times_gr.append(time_gr)
    timer.time()
    if time_gr > 10:
      flag_gr = False

  if flag_bf:
    path_bf = bellfo(graph,start,target)
    time_bf = timer.time()
    cost_bf = sum([graph.vs(v)["height"][0] for v in path_bf])
    log.debug("Bellman-Ford: " + str(time_bf) + "s, path cost: " + str(cost_bf))
    dist_bf.append(cost_bf)
    times_bf.append(time_bf)
    timer.time()
    if time_bf > 10:
      flag_bf = False

  if flag_dj:
    path_dj = dijkstra(graph,start,target)
    time_dj = timer.time()
    cost_dj = sum([graph.vs(v)["height"][0] for v in path_dj])
    log.debug("Dijkstra: " + str(time_dj) + "s, path cost: " + str(cost_dj))
    dist_dj.append(cost_dj)
    times_dj.append(time_dj)
    timer.time()
    if time_dj > 10:
      flag_dj = False
    
  if flag_as:
    path_as = Astar(graph,start,target)
    time_as = timer.time()
    cost_as = sum([graph.vs(v)["height"][0] for v in path_as])
    log.debug("A*: " + str(time_as) + "s, path cost: " + str(cost_as))
    dist_as.append(cost_as)
    times_as.append(time_as)
    timer.time()
    if time_as > 10:
      flag_as = False

  width = width + i%2
  height = height + (i+1)%2

data = [dist_ig,  dist_gr,  dist_bf,  dist_dj,  dist_as,
        times_ig,  times_gr,  times_bf,  times_dj,  times_as]
with open('data/stat.json', 'w') as outfile:
  json.dump(data, outfile)