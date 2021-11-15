from random import choices
import logging as log
import igraph as ig
from time import perf_counter as time
import numpy as np
from math import floor 
from json import loads as jload
## Graph Functions ##

def generate_edge_list(width, height):
  edges = []
  for h in range(height - 1):
    for w in range(width - 1):
      vid = wh2vid(w, h, width)
      edges.append((vid,vid + 1))
      edges.append((vid,vid + width))
    vid += 1
    edges.append((vid, vid + width))
  for w in range(width-1):
    vid = width*height - width + w
    edges.append((vid, vid + 1))
  
  return edges

def generate_graph(width, height):
  vnum = int(width*height)
  graph = ig.Graph(vnum)

  edges = generate_edge_list(width, height)
  graph.add_edges(edges)

  graph["width"] = width

  # Initial value of pheromone on each edge 
  for e in graph.es():
    e["pheromone"] = 1

  return graph

def draw_line(x0, y0, x1, y1):
  line=[(x0, y0)]
  dx = np.abs(x1-x0)
  sx = 1 if x0 < x1 else -1
  dy = -np.abs(y1-y0)
  sy = 1 if y0 < y1 else -1
  err = dx + dy

  while not ((x0 == x1) & (y0 == y1)):
    e2 = err * 2

    if e2 > dy:
      err += dy
      x0 += sx
      line.append((x0,y0))
    elif e2 < dx:
      err += dx
      y0 += sy
      line.append((x0,y0))
    # print (x0,y0)
  return line

## Helper functions ##

def man_dist(start, goal, width):
  start_x, start_y = vid2wh(start, width)
  goal_x, goal_y = vid2wh(goal, width)
  return abs(start_x - goal_x) + abs(start_y - goal_y)

def diag_dist(start, goal, width):
  start_x, start_y = vid2wh(start, width)
  goal_x, goal_y = vid2wh(goal, width)
  dx = abs(start_x - goal_x)
  dy = abs(start_y - goal_y)
  return (dx + dy) - 0.585786* min(dx, dy) 
  # (D2 - 2 * D) * min(dx, dy) - There are min(dx, dy) diagonal steps, and each one costs D2 but saves you 2x D non-diagonal steps.

def eucl_dist(start, goal, width):
  return np.linalg.norm(np.array(vid2wh(start, width)) - np.array(vid2wh(goal, width)))

def vid2wh(v, width):
  w = (v % width)
  h = int(floor(v/width))
  return w,h

def wh2vid(w, h, width):
    vid = h * width + w
    return int(vid)

def setsub(l1, l2):
  # a function for subtracting lists like sets
  return list(set(l1) - set(l2))

def update_frame(width, step, v, frame, colour):
  w, h = [x * step for x in vid2wh(v, width)]
  if colour == 'r': 
    frame[h:h+step,w:w+step,2] = 255
  elif colour == 'g':
    frame[h:h+step,w:w+step,1] = 255
  elif colour == 'b':
    frame[h:h+step,w:w+step,0] = 255
  elif colour == 'w':
    frame[h:h+step,w:w+step] = 255
  elif colour == 'k':
    frame[h:h+step,w:w+step] = 0

class Timer:
    def __init__(self):
        self.start = time()

    def time(self):
      elapsed = time() - self.start
      log.debug( f"Elapsed time: {elapsed:0.4f} seconds" )
      self.start = time()
      return elapsed

def lprint(message: str):
  print(message)
  log.info(message)

## Pathfinding functions ##

def bellfo(g, start, end):
  vn = g.vcount()
  dist = [float("inf")] * vn
  predecessor = [[]] * vn 

  dist[start] = 0

  for i in range(vn - 1):
    for e in g.es():
      u = e.source
      v = e.target
      if dist[u] + e["weight"] < dist[v]:
        dist[v] = dist[u] + e["weight"]
        predecessor[v] = u
      elif dist[v] + e["weight"] < dist[u]:
        dist[u] = dist[v] + e["weight"]
        predecessor[u] = v
  
  path = [end]
  while path[-1] != start:
    path.append(predecessor[path[-1]])
  path.reverse()
  return path


def dijkstra(g, start, end):
  vn = g.vcount()
  dist = [float("inf")] * vn
  previus = [[]] * vn
  # for v in g.vs():
  #   dist.append(float('inf'))
  #   previus.append([])

  Q = list(range(vn))
  for v in jload(g["deleted_vs"]):
    Q.remove(v)
  
  dist[start] = 0

  while len(Q) > 0:
    tmp_dist = []
    for q in range(vn):
      if q in Q:
        tmp_dist.append(dist[q])
      else:
        tmp_dist.append(float('inf'))
    
    u = tmp_dist.index(min(tmp_dist))
    Q.remove(u)
    for v in g.neighbors(u):
      eid = g.get_eid(u, v)
      alt = dist[u] + g.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        previus[v] = u

  return reconstruct_path(start, end, previus)


def reconstruct_path(start, end, previus):
  path = [end]
  while path[-1] != start:
    path.append(previus[path[-1]])
  path.reverse()
  return path

def Astar(g, start, end):
  """[summary]

  Args:
      g ([type]): [description]
      start ([type]): [description]
      end ([type]): [description]

  Returns:
      [type]: [description]
  """
  vn = g.vcount()
  p = 2/vn
  
  previus = [[]] * vn

  Q = [start]
  
  dist = [float("inf")] * vn
  dist[start] = 0

  fscore = [float("inf")] * vn
  fscore[start] = diag_dist(start, end, g["width"]) * (1+p)

  while len(Q) > 0:
    tmp_dist = []
    for q in range(vn):
      if q in Q:
        tmp_dist.append(fscore[q])
      else:
        tmp_dist.append(float('inf'))
    u = tmp_dist.index(min(tmp_dist))
    
    if u == end:
      return reconstruct_path(start, u, previus)

    Q.remove(u)
    for v in g.neighbors(u):
      eid = g.get_eid(u, v)
      alt = dist[u] + g.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        fscore[v] = alt + diag_dist(v, end, g["width"]) * (1+p)
        previus[v] = u
        if v not in Q:
          Q.append(v)


def bestfirst(g, start, end):
  vn = g.vcount()
  previus = [[]] * vn

  opened_list = [start]
  
  dist = [float("inf")] * vn
  dist[start] = 0
  
  while len(opened_list) > 0:
    tmp_dist = []
    for q in range(vn):
      if q in opened_list:
        tmp_dist.append(dist[q])
      else:
        tmp_dist.append(float('inf'))
    u = tmp_dist.index(min(tmp_dist))
    
    if u == end:
      return reconstruct_path(start, u, previus)

    opened_list.remove(u)
    for v in g.neighbors(u):
      alt = diag_dist(v, end, g["width"])
      if alt < dist[v]:
        dist[v] = alt
        previus[v] = u
        if v not in opened_list:
          opened_list.append(v)


class Ant:
  def __init__(self, _id, start):
    self.id = _id
    self.visited = [start]
    self.position = start
    self.weight_sum = 0

def antss(g: ig.Graph, start: int, end: int, 
          number_of_generations=20, number_of_ants=50,
          ph_evap_coef=0.1, ph_deposition=5,
          ph_influence=1, weight_influence=1):
  ''' 
  This function implements the Ant colony algorithm inspired by Ants
    Each generation some number of ants is released. 
    Each ant tries to reach the goal by following the pheromone

  Parameters: 
    number_of_generations - how many generations of ants there will be
    number_of_ants - number of ants in each generation
    ph_evap_coef - Pheromone evaporation coefficent - proportion of how much pheromone will evaporate each generation 
    ph_deposition - ammount of pheromone deposited for each transition
    ph_influence - parameter to control the influence of pheromone in probabilty distribution 
    weight_influence - parameter to control the influence of weight in probabilty distribution 
  '''

  log.info("weight_influence = " + str(weight_influence))
 
  # Initial value of pheromone on each edge 
  for e in g.es():
    e["pheromone"] = 1

  for generation in range(number_of_generations):
    all_paths, all_paths_weight = ant_edge_selection(g, start, end, number_of_ants, ph_influence, weight_influence)
      
    ## Pheromone update after each generation
    g = pheromone_update(g, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)

    # print(all_ways)
  
  final_path = [start]
  visited = [start]
  current = start
  while final_path[-1] != end:
    possible_ways = setsub(g.neighbors(current), visited)

    if len(possible_ways) == 0:
      log.error("Ants got lost! No way from start to end.")
      return([])

    way_distribiution = [g.es(g.get_eid(current, w))["pheromone"][0] for w in possible_ways]
    visited.append(current)
    current = possible_ways[way_distribiution.index(max(way_distribiution))]
    final_path.append(current)
  
  return(final_path)

def ant_edge_selection(g, start, end, number_of_ants, ph_influence, weight_influence):
  all_paths = []
  all_paths_weight = []
  ants = [Ant(id, start) for id in range(number_of_ants)]
  while len(ants) > 0:
    ## Edge selection for each ant 
    for ant in ants:
      # log.info("Generaion: " + str(generation) + " Ant: " + str(ant.id) + " Position: " + str(ant.position))
      possible_ways = setsub(g.neighbors(ant.position), ant.visited) # ant can go to vertice that have not been in before
      # log.debug("Posible ways: " + str(possible_ways))

      if len(possible_ways) == 0:
        log.debug(f"Ant {ant.id} got lost. Path: " + str(ant.visited))
        ants.remove(ant) # if ant is lost remove it from list
        continue
      
      # calculation of probability of each way
      ph_sum = 0
      way_distribiution_temp = []
      for way in possible_ways:
        ph = g.es(g.get_eid(ant.position, way))["pheromone"][0] ** ph_influence
        weight = g.vs[way]["distance"] ** weight_influence
        ph_sum += ph/weight

        way_distribiution_temp.append(ph/weight)
      way_distribiution = [w / ph_sum for w in way_distribiution_temp]

      # ph_sum = sum([ g.es(g.get_eid(ant.position, w))["pheromone"][0] for w in possible_ways ])
      # way_distribiution = [ (g.es(g.get_eid(ant.position, w))["pheromone"][0] / g.es(g.get_eid(ant.position, w))["weight"][0] ) / ph_sum for w in possible_ways ]
      the_way = choices(possible_ways, way_distribiution)[0] # a way chosen by ant based on pheromone and weight 
      ant.weight_sum = ant.weight_sum + g.vs[the_way]["distance"]
      ant.visited.append(the_way)
      ant.position = the_way

      # if ant reached goal, remove it from set and remember the path
      if the_way == end:
        log.debug(f"Ant {ant.id} found a way")
        all_paths.append(ant.visited)
        all_paths_weight.append(ant.weight_sum)
        ants.remove(ant)
  return all_paths,all_paths_weight

def pheromone_update(g, ph_evap_coef, ph_deposition, all_paths, all_paths_weight):
  for e in g.es():
    e["ph_update"] = 0

  for path, weight in zip(all_paths, all_paths_weight):
    length = len(path)
    for i in range(length - 1):
      hop = path[i]
      next_hop = path[i+1]
      # g.es(x)["y"] returs one element list so You need to use [0] but in order to write to it You can't do that. It's stupid
      # g.es(g.get_eid(hop, next_hop))["pheromone"] = g.es(g.get_eid(hop, next_hop))["pheromone"][0] * (1 - ph_evap_coef) + ph_deposition/weight
      g.es(g.get_eid(hop, next_hop))["ph_update"] = g.es(g.get_eid(hop, next_hop))["ph_update"][0] + ph_deposition/weight
  
  for e in g.es():
    e["pheromone"] = e["pheromone"] * (1 - ph_evap_coef) + e["ph_update"]
  
  return g
