import multiprocessing
import logging as log
from spmodule.splib import *
from json import dump
from datetime import datetime

def test(graph, start, end, costs, times, gnum, algorithm, heuristic = None):
  timer = Timer()
  if heuristic is not None:
    path = algorithm(graph, start, end, heuristic)
  else:
    path = algorithm(graph, start, end)
  atime = timer.time()
  
  cost = path_cost(graph, path)
  costs[gnum] = cost
  times[gnum] = atime
  print(f"Num: {gnum}, cost: {cost}, time: {atime}")

if __name__ == '__main__':

  log.basicConfig(level=log.WARNING,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')

  pool = multiprocessing.Pool(8)
  manager = multiprocessing.Manager()
  
  size = 100
  offset = 500
  N = 500

  costs = manager.dict()
  times = manager.dict()
  all_dict = {0: {"cost": 0, "time": 0}}
  
  algorithm = Astar
  
  heuristic = diag_dist_k
  
  params = {'size': size, "Graphs": N, "offset": offset, 
            "algorithm": algorithm.__name__, "heuristic": "diag_dist2"}
  
  for gnum in range(offset, offset + N):
    graph = ig.load(f"graphs/simulations/{size}x{size}/graph_{gnum}.graphml")
    start = int(graph["start"])
    end = int(graph["end"])
    
    # pool.apply_async(test, (graph, start, end, costs, times, gnum, algorithm, heuristic, k))
    pool.apply_async(test, (graph, start, end, costs, times, gnum, algorithm))
    
  pool.close()
  pool.join()
  
  mean_time = 0
  mean_cost = 0
  for i in range(offset, offset + N):
    mean_time += times[i]
    mean_cost += costs[i]
    all_dict[i] = {"cost": costs[i], "time": times[i]}
  
  mean_time = mean_time/N
  mean_cost = mean_cost/N
  
  print(f"Algorithm {algorithm.__name__} - Mean cost: {mean_cost}, mean time: {mean_time}")
  
  jdict = {"params": params, 
           "meantime" : mean_time, "mean_cost": mean_cost,
           "all_dict": all_dict}
  
  fname = "dumps/final/" + algorithm.__name__ + "-" + str(size) + "--" + str(datetime.now().strftime('%d-%m--%H-%M-%S')) + ".json"
  with open(fname, 'w') as f:
    dump(dict(jdict), f, indent=2)