import multiprocessing
import igraph as ig 
import logging as log
from spmodule.splib import *
from json import dump
from datetime import datetime
import multiprocessing

def test(graph, start, end, costs, times, gnum, k):
  timer = Timer()

  path = Astar_heuristic(graph, start, end, diag_dist_k , k/5)
  
  atime = timer.time()
  cost = path_cost(graph, path)
  costs[f"{gnum}:{k}"] = cost
  times[f"{gnum}:{k}"] = atime
  print(f"Num: {gnum}, k: {k}, cost: {cost}, time: {atime}")
  
  
if __name__ == '__main__':

  full_timer = Timer()
  
  log.basicConfig(level=log.INFO,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')

  manager = multiprocessing.Manager()
  
  size = 100
  offset = 500
  N = 500
  test_range = range(5,51,1)

  costs = manager.dict()
  times = manager.dict()
  sum_dict = {0: {"cost": 0, "time": 0}}
  allgen_dict = {0: {0: {"cost": 0, "time": 0}}}
  params = {'size': size, "Graphs": N, "offset": offset, 
            "range": [test_range[0],test_range[-1], test_range[1]-test_range[0]]}
  
  for ngen in test_range:
    sum_dict[ngen] = {"cost": 0, "time": 0}
    allgen_dict[ngen] = {0: {"cost": 0, "time": 0}}
    
  pool = multiprocessing.Pool(8)
  
  
  for graph_num in range(offset, offset + N):
    graph = ig.load(f"graphs/simulations/{size}x{size}/graph_{graph_num}.graphml")
    start = int(graph["start"])
    end = int(graph["end"])
    
    for k in test_range:
      pool.apply_async(test, (graph, start, end, costs, times, graph_num, k))
      
      
  pool.close()
  pool.join()
  for i in range(offset, offset + N):
    for value in test_range:
      allgen_dict[value][i] = {"cost": costs[f"{i}:{value}"], "time": times[f"{i}:{value}"]}
      sum_dict[value]['cost'] += costs[f"{i}:{value}"]
      sum_dict[value]['time'] += times[f"{i}:{value}"]
      
  print(sum_dict)        
  
  jdict = {"params": params, "time": full_timer.time(), "sum_dict" : sum_dict, "allgen_dict": allgen_dict}

  # print(mean_costs)
  fname = "dumps/astar_hk-" + str(datetime.now().strftime('%d-%m--%H-%M-%S')) + ".json"
  with open(fname, 'w') as f:
    dump(dict(jdict), f, indent=2)