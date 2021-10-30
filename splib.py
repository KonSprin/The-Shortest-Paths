from random import choices
import logging as log
from collections import deque
import igraph as ig
from time import perf_counter as time

class Timer:
    def __init__(self):
        self.start = time()

    def time(self):
      elapsed = time() - self.start
      log.debug( f"Elapsed time: {elapsed:0.4f} seconds" )
      self.start = time()
      return elapsed

def setsub(l1, l2):
  # a function for subtracting lists like sets
  return list(set(l1) - set(l2))

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
  return [path]


def dijkstra(g, start, end):
  vn = g.vcount()
  dist = [float("inf")] * vn
  previus = [[]] * vn
  # for v in g.vs():
  #   dist.append(float('inf'))
  #   previus.append([])

  Q = list(range(vn))
  
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

  path = [end]
  while path[-1] != start:
    path.append(previus[path[-1]])
  path.reverse()
  return [path]

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
    all_paths, all_paths_weight = ant_edge_selection(g, start, end, number_of_ants, ph_influence, weight_influence, generation)
      
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
