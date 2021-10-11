from random import choices
import logging as log
from collections import deque

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
      alt = dist[u] + g.es(g.get_eid(u, v))["weight"][0]
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


def antss(g, start, end, g_num):
  ant_num = 10 # number of ants in each generation

  for e in g.es():
    e["pheromone"] = 10

  p = 0.1 # Pheromone evaporation coefficent
  T = 5 # ammount of pheromone deposited for each transition

  for generation in range(g_num):
    all_ways = deque() # Using deque insted of normal list becouse prepending to list is super inefficient in large lists
    ants = [Ant(id, start) for id in range(ant_num)]
    while len(ants) > 0:
      # Edge selection for each ant 
      for ant in ants:
        log.info("Generaion: " + str(generation) + " Ant: " + str(ant.id) + " Position: " + str(ant.position))
        possible_ways = setsub(g.neighbors(ant.position), ant.visited) # ant can go to vertice that have not been in before
        log.debug("Posible ways: " + str(possible_ways))

        if len(possible_ways) == 0:
          ants.remove(ant) # if ant is lost remove it from list
          continue
        
        # calculation of probability of each way
        ph_sum = sum([ g.es(g.get_eid(ant.position, w))["pheromone"][0] for w in possible_ways ])
        w_dist = [ (g.es(g.get_eid(ant.position, w))["pheromone"][0] / g.es(g.get_eid(ant.position, w))["weight"][0] ) / ph_sum for w in possible_ways ]
        the_way = choices(possible_ways, w_dist)[0]
        ant.visited.append(the_way)
        ant.position = the_way

        # if ant reached goal, remove it from set and remember the path
        if the_way == end:
          all_ways.append(deque(ant.visited))
          ants.remove(ant)
      
    # Pheromone update after each generation
    for way in all_ways:
      for i in range(len(way) - 1):
        hop = way[i]
        next_hop = way[i+1]
        # g.es(x)["y"] returs one element list so You need to use [0] but in order to write to it You can't do that. It's stupid
        g.es(g.get_eid(way[i], way[i+1]))["pheromone"] = g.es(g.get_eid(way[i], way[i+1]))["pheromone"][0] * (1 - p) + T

    # print(all_ways)
  
  final_way = [start]
  visited = [start]
  current = start
  while final_way[-1] != end:
    possible_ways = setsub(g.neighbors(current), visited)

    if len(possible_ways) == 0:
      log.error("Ants got lost! No way from start to end.")
      return([])

    w_dist = [g.es(g.get_eid(current, w))["pheromone"][0] for w in possible_ways]
    visited.append(current)
    current = possible_ways[w_dist.index(max(w_dist))]
    final_way.append(current)
  
  return(final_way)







