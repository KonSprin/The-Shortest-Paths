import logging as log
import igraph as ig
import numpy as np
import noise
from time import perf_counter as time
from random import choices
from math import floor 
from random import random, randint, sample

## Graph Functions ##

def generate_edge_list(width, height):
  edges = []
  for h in range(height - 1):
    for w in range(width - 1):
      vid = wh2vid(w, h, width)
      edges.append((vid,vid + 1))
      edges.append((vid,vid + width))
      edges.append((vid,vid + width + 1))
      edges.append((vid + 1,vid + width))
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
    dd = abs(e.target - e.source)
    if (dd == 1) or (dd ==width):
      e["weight"] = 1
    else:
      e["weight"] = 1.4142
      
  
  for v in graph.vs():
    v["height"] = 1

  return graph

def generate_weighted_graph(width, height, step, start, end, mountain_height, wall_percent):
  frame_width = width * step
  frame_height = height * step
  path = [[]]
  while path == [[]]:
    graph = generate_graph(width, height)
    img = np.zeros((frame_height,frame_width,3), np.uint32)
    # add_mountains(graph, img, sample(range(width*height), no_mountains), mountain_height, step)
    add_perlin_mountains(graph,img,mountain_height, step, width/3)

    random_points(graph, img, step, wall_percent, start, end)
    try: 
      path = graph.get_shortest_paths(start, end)
    except RuntimeWarning: 
      log.warning("lmao")
      pass
  update_frame(width, step, end, img, 'b')
  update_frame(width, step, start, img, 'b')
  return graph,img

def generate_weighted_graph_noimg(width, height, start, end, mountain_height, wall_percent):
  path = [[]]
  while path == [[]]:
    graph = generate_graph(width, height)
    
    add_perlin_mountains_noimg(graph, mountain_height, width/3)

    random_points_noimg(graph, wall_percent, start, end)
    try: 
      path = graph.get_shortest_paths(start, end)
    except RuntimeWarning: 
      log.warning("lmao")
      pass

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
  if dx > dy:
    return 0.4*dy + dx
  return 0.4*dx + dy

def diag_dist_k(start, goal, width, k):
  start_x, start_y = vid2wh(start, width)
  goal_x, goal_y = vid2wh(goal, width)
  dx = abs(start_x - goal_x)
  dy = abs(start_y - goal_y)
  if dx > dy:
    return (0.4*dy + dx) * k
  return (0.4*dx + dy) * k

def diag_dist10(start, goal, width):
  start_x, start_y = vid2wh(start, width)
  goal_x, goal_y = vid2wh(goal, width)
  dx = abs(start_x - goal_x)
  dy = abs(start_y - goal_y)
  if dx > dy:
    return 14*dy + 10*(dx-dy)
  return 14*dx + 10*(dy-dx)

def zero(start, goal, width):
    return 0

def octile_dist(start, goal, width):
  start_x, start_y = vid2wh(start, width)
  goal_x, goal_y = vid2wh(goal, width)
  dx = abs(start_x - goal_x)
  dy = abs(start_y - goal_y)
  return ((dx + dy) - 0.585786* min(dx, dy)) * 5
  # (D2 - 2 * D) * min(dx, dy) - There are min(dx, dy) diagonal steps, and each one costs D2 but saves you 2x D non-diagonal steps.

def eucl_dist(start, goal, width):
  return np.linalg.norm(np.array(vid2wh(start, width)) - np.array(vid2wh(goal, width)))

def vid2wh(v, width):
  w = int(v % width)
  h = int(floor(v/width))
  return w,h

def wh2vid(w, h, width):
    vid = h * width + w
    return int(vid)

def setsub(l1, l2):
  # a function for subtracting lists like sets
  return list(set(l1) - set(l2))

def update_frame(width, step, v, frame, colour = 'rand'):
  w, h = [x * step for x in vid2wh(v, width)]
  if colour == 'r': 
    frame[h:h+step,w:w+step] = 0
    frame[h:h+step,w:w+step,2] = 255
  elif colour == 'g':
    frame[h:h+step,w:w+step] = 0
    frame[h:h+step,w:w+step,1] = 255
  elif colour == 'b':
    frame[h:h+step,w:w+step] = 0
    frame[h:h+step,w:w+step,0] = 255
  elif colour == 'w':
    frame[h:h+step,w:w+step] = 255
  elif colour == 'k':
    frame[h:h+step,w:w+step] = 0
  elif colour == 'rand':
    frame[h:h+step,w:w+step,0] = randint(0,255)
    frame[h:h+step,w:w+step,1] = randint(0,255)
    frame[h:h+step,w:w+step,2] = randint(0,255)

def random_points(graph, img, step, percentage, start, target):
  vcount = graph.vcount()
  
  v = choose_v(vcount, [start, target])
      
  deleted = []  
  while len(deleted) < (vcount * percentage / 100):
    v_nei = graph.neighbors(v)
    
    if decision(0.1) or len(v_nei) == 0:
      v = choose_v(vcount, [start, target] + deleted)
      continue
    
    graph.delete_edges(graph.incident(v))
    deleted.append(v)
    for nei in v_nei:
      if nei in [start, target]:
        v_nei.remove(nei)
    if len(v_nei) == 0:
      v = choose_v(vcount, [start, target] + deleted)
      continue
    v = v_nei[randint(0, len(v_nei)-1)]

  for v in deleted:
    w, h = vid2wh(v, graph["width"])
    img[h*step:h*step+step,w*step:w*step+step,:] = 0
  
def random_points_noimg(graph, percentage, start, target):
  vcount = graph.vcount()
  
  v = choose_v(vcount, [start, target])
      
  deleted = []  
  while len(deleted) < (vcount * percentage / 100):
    v_nei = graph.neighbors(v)
    
    if decision(0.1) or len(v_nei) == 0:
      v = choose_v(vcount, [start, target] + deleted)
      continue
    
    graph.delete_edges(graph.incident(v))
    deleted.append(v)
    for nei in v_nei:
      if nei in [start, target]:
        v_nei.remove(nei)
    if len(v_nei) == 0:
      v = choose_v(vcount, [start, target] + deleted)
      continue
    v = v_nei[randint(0, len(v_nei)-1)]


def decision(probability):
  return random() < probability
  
def choose_v(vcount, exclude):
    v = None
    while v is None:
      v = randint(0, vcount-1)
      if v in exclude:
        v = None
    return v
  
def smoothen_terrain(graph):
  height = []
  
  for v in graph.vs():
    height.append(v["height"])

  smothened = []

  while len(smothened) < graph.vcount():
    
    tmp_height = height
    for u in smothened:
      tmp_height[u] = 0
    v = tmp_height.index(max(tmp_height))
    
    smothened.append(v)
    v_height = graph.vs(v)["height"][0] * 0.9
    for nei in graph.neighbors(v):
      vnei = graph.vs(nei)
      if vnei["height"][0] < v_height:
        height[nei] = v_height
        vnei["height"] = v_height
 
def add_mountains(graph, img, mountain_list, height, step):  
  for mountain in mountain_list:
    graph.vs(mountain)["height"] = height
    
  smoothen_terrain(graph)
  
  for e in graph.es():
    e["weight"] = e["weight"] * (graph.vs(e.target)["height"][0] + graph.vs(e.source)["height"][0]) / 2
  
  scalar = 100/height
  for v in graph.vs():
    w, h = vid2wh(v.index, graph["width"])
    img[h*step:h*step+step,w*step:w*step+step,2] = v["height"] * scalar

def add_perlin_mountains(graph, img, height, step, scale = 30.0, octaves = 6, persistence = 0.5, lacunarity = 2.0):
  width = graph["width"]
  shape = (int(graph.vcount()/width), width)
  img[:,:,2] = 255
  base = int(randint(1,300))
  for i in range(shape[0]):
    for j in range(shape[1]):
      nois = 0
      for l in [1, 2, 4, 8]:
        nois += noise.pnoise2(i/(scale/l), j/(scale/l), octaves=octaves, 
                             persistence=persistence, lacunarity=lacunarity, 
                             repeatx=1024, repeaty=1024, base=base)/l
      img[i*step:i*step+step,j*step:j*step+step,0:2] = 255 - np.uint32((np.tanh(nois)/2 + 0.5) * 255)
      graph.vs(wh2vid(j,i,width))["height"] = (np.tanh(nois)/2 + 0.5) * height
  
  for e in graph.es():
    e["weight"] = e["weight"] * (graph.vs(e.target)["height"][0] + graph.vs(e.source)["height"][0]) / 2
 
def add_perlin_mountains_noimg(graph, height, scale = 30.0, octaves = 6, persistence = 0.5, lacunarity = 2.0):
  width = graph["width"]
  shape = (int(graph.vcount()/width), width)
  base = randint(0,300)
  for i in range(shape[0]):
    for j in range(shape[1]):
      nois = 0
      for l in [1, 2, 4, 8]:
        nois += noise.pnoise2(i/(scale/l), j/(scale/l), octaves=octaves, 
                             persistence=persistence, lacunarity=lacunarity, 
                             repeatx=1024, repeaty=1024, base=base)/l
      graph.vs(wh2vid(j,i,width))["height"] = (np.tanh(nois)/2 + 0.5) * height
  
  for e in graph.es():
    e["weight"] = e["weight"] * (graph.vs(e.target)["height"][0] + graph.vs(e.source)["height"][0]) / 2

def add_mountains_noimg(graph, mountain_list, height):  
  for mountain in mountain_list:
    graph.vs(mountain)["height"] = height
    
  smoothen_terrain(graph)
  
  for e in graph.es():
    e["weight"] = e["weight"] * (graph.vs(e.target)["height"][0] + graph.vs(e.source)["height"][0]) / 2

   
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


def dijkstra(graph, start, end):
  vn = graph.vcount()
  dist = [float("inf")] * vn
  previus = [[]] * vn

  opened = list(range(vn))
  
  dist[start] = 0
  closed = []

  while len(opened) > 0:
    tmp_dist = dist.copy()
    for q in closed:
      tmp_dist[q] = float('inf')
      
    u = tmp_dist.index(min(tmp_dist))
    opened.remove(u)
    closed.append(u)
    
    if u == end: return reconstruct_path(start, end, previus)
    
    for v in graph.neighbors(u):
      eid = graph.get_eid(u, v)
      alt = dist[u] + graph.es(eid)["weight"][0]
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

  opened = [start]
  closed = []
  
  dist = [float("inf")] * vn
  dist[start] = 0

  fscore = [float("inf")] * vn
  fscore[start] = diag_dist(start, end, g["width"]) * (1+p)

  while len(opened) > 0:
    # tmp_dist = fscore.copy()
    # for q in closed:
    #   tmp_dist[q] = float('inf')

    tmp_dist = [float("inf")] * vn
    for q in opened:
      tmp_dist[q] = fscore[q]
    u = tmp_dist.index(min(tmp_dist))
    
    if u == end:
      return reconstruct_path(start, u, previus)

    closed.append(u)
    opened.remove(u)
    for v in g.neighbors(u):
      eid = g.get_eid(u, v)
      alt = dist[u] + g.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        fscore[v] = alt + diag_dist(v, end, g["width"]) * (1+p)
        previus[v] = u
        if v not in opened:
          opened.append(v)

def Astar_heuristic(g, start, end, heuristic, k = None):
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

  opened = [start]
  closed = []
  
  dist = [float("inf")] * vn
  dist[start] = 0

  fscore = [float("inf")] * vn
  if k is not None:
    fscore[start] = heuristic(start, end, g["width"], k) * (1+p)
  else: 
    fscore[start] = heuristic(start, end, g["width"]) * (1+p)
  while len(opened) > 0:
    # tmp_dist = fscore.copy()
    # for q in closed:
    #   tmp_dist[q] = float('inf')

    tmp_dist = [float("inf")] * vn
    for q in opened:
      tmp_dist[q] = fscore[q]
    u = tmp_dist.index(min(tmp_dist))
    
    if u == end:
      return reconstruct_path(start, u, previus)

    closed.append(u)
    opened.remove(u)
    for v in g.neighbors(u):
      eid = g.get_eid(u, v)
      alt = dist[u] + g.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        if k is not None:
          fscore[v] = alt + heuristic(v, end, g["width"], k) * (1+p)
        else:
          fscore[v] = alt + heuristic(v, end, g["width"]) * (1+p)
        previus[v] = u
        if v not in opened:
          opened.append(v)
          
def bestfirst(g, start, end):
  vn = g.vcount()
  previus = [[]] * vn

  opened_list = [start]
  
  dist = [float("inf")] * vn
  dist[start] = 0
  
  while len(opened_list) > 0:
    tmp_dist = []
    for opened in range(vn):
      if opened in opened_list:
        tmp_dist.append(dist[opened])
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
    self.deadlocked = []

def antss(g: ig.Graph, start: int, end: int, 
          number_of_generations=5, number_of_ants=20,
          ph_evap_coef=0.20, 
          ph_influence=1, weight_influence=2, visibility_influence=1):
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

  ph_deposition = g.ecount() * ph_evap_coef
  
  width = g["width"]
  for v in g.vs():
    v["distance"] = diag_dist(v.index, end, width)
  g.vs[end]["distance"] = 0.0001
  
  log.info("weight_influence = " + str(weight_influence))
 
  # Initial value of pheromone on each edge 
  for e in g.es():
    e["pheromone"] = 1
    
  # path_grd = bestfirst(g,start,end)
  # pr = path_grd[0]
  # for v in path_grd[1:]:
  #   g.es(g.get_eid(pr,v))["pheromone"] = 40
  #   pr = v
    
  
  best_gen_path = []
  best_gen_path_weight = []
  for generation in range(number_of_generations):
    all_paths, all_paths_weight = ant_edge_selection(g, start, end, number_of_ants, ph_influence, weight_influence, visibility_influence)
    
    
    # if len(all_paths) > 0:
    best_gen_path_index = all_paths_weight.index(min(all_paths_weight))
    best_gen_path.append(all_paths[best_gen_path_index])
    best_gen_path_weight.append(all_paths_weight[best_gen_path_index])
    log.debug(f"Ended generation {generation} with best paths cost: {best_gen_path_weight[-1]}")
      
    # lprint(f"Ended generation {generation} with best paths cost: {best_gen_path_weight[-1]}")    
    
    ## Pheromone update after each generation
    # g = pheromone_update(g, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)
    g = minmax_pheromone_update(g, ph_evap_coef, ph_deposition, best_gen_path[-1], best_gen_path_weight[-1])
    
    # ph_influence=ph_influence*1.1
    # weight_influence=weight_influence*0.9
    
    # if len(all_paths) > 0:
    #   best_gen_path_index = all_paths_weight.index(min(all_paths_weight))
    #   best_gen_path.append(all_paths[best_gen_path_index])
    #   best_gen_path_weight.append(all_paths_weight[best_gen_path_index])
    #   log.debug("Best path this generation: " + str(best_gen_path_weight[-1]))

    #   num_reached += len(all_paths)
    # print(all_ways)
  return best_gen_path[best_gen_path_weight.index(min(best_gen_path_weight))]

  # final_path = [start]
  # visited = [start]
  # current = start
  # while final_path[-1] != end:
  #   possible_ways = setsub(g.neighbors(current), visited)

  #   if len(possible_ways) == 0:
  #     log.error("Ants got lost! No way from start to end.")
  #     return([])

  #   way_distribiution = [g.es(g.get_eid(current, w))["pheromone"][0] for w in possible_ways]
  #   visited.append(current)
  #   current = possible_ways[way_distribiution.index(max(way_distribiution))]
  #   final_path.append(current)
  
  # return(final_path)

def ant_edge_selection(g, start, end, number_of_ants, ph_influence, weight_influence, visibility_influence):
  
  all_paths = []
  all_paths_weight = []
  ants = [Ant(id, start) for id in range(number_of_ants)]
  while len(ants) > 0:
    ## Edge selection for each ant 
    for ant in ants:
      possible_ways = setsub(g.neighbors(ant.position), ant.visited + ant.deadlocked) # ant can go to vertice that have not been in before
      # possible_ways = g.neighbors(ant.position)
      
      if len(possible_ways) == 0:
        # log.debug(f"Ant {ant.id} got lost. Path: " + str(ant.visited))
        
        # ant.weight_sum = ant.weight_sum - g.vs[ant.visited[-1]]["height"]
        ant.deadlocked.append(ant.visited.pop())
        ant.position = ant.visited[-1]
        
        # ants.remove(ant)
        continue
      elif len(possible_ways) == 1:
         the_way = possible_ways[0]
      else:
        # calculation of probability of each way
        ph_sum = 0
        way_distribiution_temp = []
        position_distance = g.vs[ant.position]["distance"]
        for way in possible_ways:
          ph = g.es(g.get_eid(ant.position, way))["pheromone"][0] ** ph_influence
          visibility = (position_distance/g.vs[way]["distance"]) ** visibility_influence
          weight =  (1/g.es(g.get_eid(ant.position, way))["weight"][0]) ** weight_influence
          ph_sum += ph * weight * visibility

          way_distribiution_temp.append(ph * weight * visibility)
        # way_distribiution = [wd - min(way_distribiution) + 1 for wd in way_distribiution] # normalize the shit out of this
        # way_distribiution = [(w - min(way_distribiution_temp)) for w in way_distribiution_temp]
        way_distribiution = [w/ph_sum for w in way_distribiution_temp]
        # way_distribiution1 = [(w/max(way_distribiution_temp))/ph_sum for w in way_distribiution_temp]
        # way_distribiution2 = [(w - min(way_distribiution_temp)*0.99)/max(way_distribiution_temp)/ph_sum for w in way_distribiution_temp]
        # way_distribiution = [(w - min(way_distribiution_temp))/ph_sum for w in way_distribiution_temp]
        the_way = choices(possible_ways, way_distribiution)[0] # a way chosen by ant based on pheromone and weight 
        # ph_sum = sum([ g.es(g.get_eid(ant.position, w))["pheromone"][0] for w in possible_ways ])
        # way_distribiution = [ (g.es(g.get_eid(ant.position, w))["pheromone"][0] / g.es(g.get_eid(ant.position, w))["weight"][0] ) / ph_sum for w in possible_ways ]
      
      # ant.weight_sum = ant.weight_sum + g.vs[the_way]["height"]
      ant.visited.append(the_way)
      ant.position = the_way

      # if ant reached goal, remove it from set and remember the path
      if the_way == end:
        log.debug(f"Ant {ant.id} found a way")
        path = remove_loops(g, ant.visited)
        # path = ant.visited
        all_paths.append(path)
        all_paths_weight.append(path_cost(g, path))
        ants.remove(ant)
        
  return all_paths,all_paths_weight

def pheromone_update(g, ph_evap_coef, ph_deposition, all_paths, all_paths_weight):
  for e in g.es():
    e["ph_update"] = 0

  for path, weight in zip(all_paths, all_paths_weight):
    short_path = remove_loops(g, path)
    length = len(short_path)
    for i in range(length - 1):
      hop = short_path[i]
      next_hop = short_path[i+1]
      # g.es(x)["y"] returs one element list so You need to use [0] but in order to write to it You can't do that. It's stupid
      # g.es(g.get_eid(hop, next_hop))["pheromone"] = g.es(g.get_eid(hop, next_hop))["pheromone"][0] * (1 - ph_evap_coef) + ph_deposition/weight
      g.es(g.get_eid(hop, next_hop))["ph_update"] = g.es(g.get_eid(hop, next_hop))["ph_update"][0] + ph_deposition/weight
  
  for e in g.es():
    e["pheromone"] = e["pheromone"] * (1 - ph_evap_coef) + e["ph_update"]
  
  return g

def minmax_pheromone_update(g, ph_evap_coef, ph_deposition, path, path_weight):
  for e in g.es():
    e["ph_update"] = 0

  length = len(path)
  for i in range(length - 1):
    hop = path[i]
    next_hop = path[i+1]
    # g.es(x)["y"] returs one element list so You need to use [0] but in order to write to it You can't do that. It's stupid
    # g.es(g.get_eid(hop, next_hop))["pheromone"] = g.es(g.get_eid(hop, next_hop))["pheromone"][0] * (1 - ph_evap_coef) + ph_deposition/weight
    g.es(g.get_eid(hop, next_hop))["ph_update"] = g.es(g.get_eid(hop, next_hop))["ph_update"][0] + ph_deposition/length

  for e in g.es():
    e["pheromone"] = e["pheromone"] * (1 - ph_evap_coef) + e["ph_update"]
  
  return g

def remove_loops(g, path):
  tmp_path = [path[0]]
  i = 1
  while tmp_path[-1] != path[-1]:
    node = path[i]
    tmp_path.append(node)
    
    possible_shortcuts = []
    for neighbor in g.neighbors(node):
      if neighbor in path[i+2:]:
        possible_shortcuts.append(path.index(neighbor))
        
    if len(possible_shortcuts) > 0:
      i = max(possible_shortcuts)
    else: 
      i += 1
      
  return tmp_path

def path_cost(graph, path):
  cost = 0
  length = len(path)
  
  for i in range(length - 1):
    hop = path[i]
    next_hop = path[i+1]
    cost += graph.es(graph.get_eid(hop, next_hop))["weight"][0]
    
  return cost
    