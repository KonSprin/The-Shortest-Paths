from random import choices
import logging as log
from collections import deque

def bellfo(g, start, end):
  n = g.vcount()
  dist = [float("inf")] * n
  predecessor = [[]] * n

  dist[start] = 0

  for i in range(g.vcount() - 1):
    for e in g.es():
       u = e.source
       v = e.target
       if dist[u] + 1 < dist[v]:
         dist[v] = dist[u] + 1
         predecessor[v] = u
  
  path = [end]
  while path[-1] != start:
    path.append(predecessor[path[-1]])
  path.reverse()
  return [path]


def dijkstra(g, start, end):
  dist = []
  previus = []
  for v in g.vs():
    dist.append(float('inf'))
    previus.append([])

  Q = list(range(g.vcount()))
  
  dist[start] = 0

  while len(Q) > 0:
    u = Q.pop(dist.index(min(dist)))
    for v in g.neighbors(u):
      alt = dist[u] + 1
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


def antss(g, start, end):
  def setsub(l1, l2):
    # a function for subtracting lists like sets
    return list(set(l1) - set(l2))

  ant_num = 10 # number of ants in each generation

  for e in g.es():
    e["pheromone"] = 10

  p = 0.1 # Pheromone evaporation coefficent
  T = 5 # ammount of pheromone deposited for each transition

  for generation in range(10):
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
        ph_sum = sum([g.es(g.get_eid(ant.position, w))["pheromone"][0] for w in possible_ways])
        w_dist = [g.es(g.get_eid(ant.position, w))["pheromone"][0]/ph_sum for w in possible_ways]
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







