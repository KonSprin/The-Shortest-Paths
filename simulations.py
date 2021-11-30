import multiprocessing
import logging as log
from spmodule.splib import *
from random import sample
from json import dump
from datetime import datetime

def algorithm_test(ig_dict, grd_dict, bf_dict, dj_dict, as_dict, ant_dict, i, params):
  
  width = params['width' ]
  height = params['height']
  
  mountain_height = params['mountain_height']
  wall_percent = params['wall_percent']
  
  number_of_generations = params['number_of_generations']
  number_of_ants = params['number_of_ants']
  weight_influence = params['weight_influence']
  
  timer = Timer()
  ftimer = Timer()
  
  start, target = sample(range(width*height - 1), 2)
  graph = generate_weighted_graph_noimg(width, height, start, target, mountain_height, wall_percent)
  lprint(f"Generated graph in {timer.time()} seconds. Iteration: " + str(i))
  
  
  path_ig = graph.get_shortest_paths(start,target)[0]
  time_ig = str(timer.time())
  # lprint("Igraph default finished")
  
  path_grd = bestfirst(graph,start,target)
  time_grd = str(timer.time())
  # lprint("Best first greedy algorithm finished")
  
  # path_bf = bellfo(graph,start,target)
  # time_bf = str(timer.time())
  # lprint("Bellman-Ford finished")
  
  path_dj = dijkstra(graph,start,target)
  time_dj = str(timer.time())
  # lprint("Dijkstra finished")
  
  path_as = Astar(graph,start,target)
  time_as = str(timer.time())
  # lprint("A* finished")
  
  path_ant = antss(graph,start,target,
                   number_of_generations=number_of_generations,number_of_ants=number_of_ants, 
                   weight_influence=weight_influence)
  time_ant = str(timer.time())
  # lprint("Ants finished")
  
         
  cost_ig = path_cost(graph, path_ig)
  # lprint("Igraph default: " + time_ig + "s, path cost: " + str(cost_ig))

  cost_grd = path_cost(graph, path_grd)
  # lprint("Best first greedy algorithm: " + time_grd + "s, path cost: " + str(cost_grd))
  
  # cost_bf = path_cost(graph, path_bf)
  # lprint("Bellman-Ford: " + time_bf + "s, path cost: " + str(cost_bf))

  cost_dj = path_cost(graph, path_dj)
  # lprint("Dijkstra: " + time_dj + "s, path cost: " + str(cost_dj))

  cost_as = path_cost(graph, path_as)
  # lprint("A*: " + time_as + "s, path cost: " + str(cost_as))

  cost_ant = path_cost(graph, path_ant)
  # lprint("Ants: " + time_ant + "s, path cost: " + str(cost_ant))
  
  
  ig_dict[i] = {'cost': cost_ig, 'time': float(time_ig)}
  grd_dict[i] = {'cost': cost_grd, 'time': float(time_grd)}
  # bf_dict[i] = {'cost': cost_bf, 'time': float(time_bf)}
  dj_dict[i] = {'cost': cost_dj, 'time': float(time_dj)}
  as_dict[i] = {'cost': cost_as, 'time': float(time_as)}
  ant_dict[i] = {'cost': cost_ant, 'time': float(time_ant)}
  lprint(f"Finished iteration number {i} in {ftimer.time()} seconds --- A*: {time_as}s ## Ants: {time_ant}s")
  

if __name__ == '__main__':

  log.basicConfig(level=log.WARNING,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')

  pool = multiprocessing.Pool(8)
  manager = multiprocessing.Manager()
  
  ig_dict = manager.dict()
  grd_dict = manager.dict()
  bf_dict = manager.dict()
  dj_dict = manager.dict()
  as_dict = manager.dict()
  ant_dict = manager.dict()
  
  test_name = "true"

  width = 100
  height = 100
  
  mountain_height = 10
  wall_percent = 5
  
  N = 1000
  
  parameters = {'width': width, 'height': height, 'N': N,
                'mountain_height': mountain_height, 'wall_percent': wall_percent,
                'number_of_generations': 5, 'number_of_ants':20, 'weight_influence': 3}

  ftimer = Timer()
    
  for i in range(N):
    try:
      pool.apply_async(algorithm_test, (ig_dict, grd_dict, bf_dict, dj_dict, as_dict, ant_dict, i, parameters))
    except ValueError:
      pass
    
  pool.close()
  pool.join()
  
  print(f"Generated tests in {ftimer.time()} seconds")
  
  sum_dict = {'igraph': {'cost': 0, 'time': 0},
              'greedy': {'cost': 0, 'time': 0},
              # 'bellfo': {'cost': 0, 'time': 0} ,
              'dijkstra': {'cost': 0, 'time': 0},
              'astar': {'cost': 0, 'time': 0},
              'ants': {'cost': 0, 'time': 0}}
  
  shared_dict = {'igraph': dict(ig_dict),
                 'greedy': dict(grd_dict),
                #  'bellfo': dict(bf_dict) ,
                 'dijkstra': dict(dj_dict),
                 'astar': dict(as_dict),
                 'ants': dict(ant_dict)}
    
  for key in sum_dict.keys():
    for i in range(N):
      sum_dict[key]['cost'] += shared_dict[key][i]['cost']
      sum_dict[key]['time'] += shared_dict[key][i]['time']
  
  print(sum_dict)
  
  jdict = {"shared_dict": shared_dict, "sum_dict" : sum_dict, "parameters": parameters}
  
  fname = "dumps/" + test_name + f"{width}x{height}x{N}-" + str(datetime.now().strftime('%d-%m--%H-%M-%S')) + ".json"
  with open(fname, 'w') as f:
    dump(jdict, f, indent=2)