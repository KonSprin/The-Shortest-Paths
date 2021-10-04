from random import choices
import logging as log

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

  for generation in range(10):
    all_ways = []
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
          all_ways.append(ant.visited)
          ants.remove(ant)
    print(all_ways)





