import multiprocessing
import igraph as ig 
import logging as log
from spmodule.splib import *
from json import dump
from datetime import datetime
import multiprocessing
from numpy import arange

def influence_test(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence):
    cost_sum = 0
    for start, end in zip(starts,ends):
      path = antss(graph, start, end, 8, 20, 0.04, ph_influence, weight_influence, visibility_influence)
      # path = graph.get_shortest_paths(start,end)[0]
      # sleep(randint(0,2))
      cost_sum += path_cost(graph, path)
          
    name = "ph:" + str(ph_influence) + ":wg:" + str(weight_influence) + ":vi:" + str(visibility_influence) + ":wdth:" + str(int(width))
    mean_costs[name] = cost_sum/len(starts)
    print(cost_sum/len(starts))
    
def count_test(graph, start, end, ant_count, costs, times, gnum):
  timer = Timer()
  path = antss(graph, start, end, number_of_generations = 5, 
              number_of_ants = ant_count, ph_evap_coef = 0.04,
              ph_influence= 1, weight_influence= 2, visibility_influence= 1)
  atime = timer.time()
        
  cost = path_cost(graph, path)
  costs[f"{gnum}:{ant_count}"] = cost
  times[f"{gnum}:{ant_count}"] = atime
  print(f"Num: {gnum}, ants: {ant_count}, cost: {cost}, time: {atime}")
    
    
def generation_test(graph, start, end, generations, costs, times, gnum):
  timer = Timer()
  path = antss(graph, start, end, number_of_generations = generations, 
                number_of_ants = 20, ph_evap_coef = 0.04,
                ph_influence= 1, weight_influence= 2, visibility_influence= 1)
  atime = timer.time()
  cost = path_cost(graph, path)
  costs[f"{gnum}:{generations}"] = cost
  times[f"{gnum}:{generations}"] = atime
  print(f"Num: {gnum}, generations: {generations}, cost: {cost}, time: {atime}")
    
def evap_test(graph, start, end, evap_coef, costs, times, gnum):
  timer = Timer()
  path = antss(graph, start, end, number_of_generations = 5, 
                number_of_ants = 20, ph_evap_coef = evap_coef,
                ph_influence= 1, weight_influence= 2, visibility_influence= 1)
  atime = timer.time()
  cost = path_cost(graph, path)
  costs[f"{gnum}:{evap_coef}"] = cost
  times[f"{gnum}:{evap_coef}"] = atime
  print(f"Num: {gnum}, evap_coef: {evap_coef}, cost: {cost}, time: {atime}")
  
if __name__ == '__main__':

  log.basicConfig(level=log.INFO,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')

  LOG = log.getLogger()

  # graphs = ["graphs/optimization100x50.graphml", "graphs/optimization60x40.graphml", "graphs/optimization30x30.graphml"]
  # starts_dict = {"graphs/optimization100x50.graphml": [0, 4508, 0, 4508, 0, 350, 350, 3501], "graphs/optimization60x40.graphml": [0, 59, 0, 59, 2399, 2301], "graphs/optimization30x30.graphml": [0, 29, 0, 29, 899, 871]}
  # ends_dict = {"graphs/optimization100x50.graphml": [4999, 99, 99, 4999, 4508, 4999, 4508, 3599],  "graphs/optimization60x40.graphml": [2399, 2301, 59, 2399, 2301, 0], "graphs/optimization30x30.graphml": [899, 871, 29, 899, 871, 0]}

  manager = multiprocessing.Manager()
  
  # elif test_name == "influences":
  #   ant_count = 20
  #   ph_evap_coef = 0.04
  #   for ph_influence in range(1,5):
  #     for weight_influence in range(1,5):
  #       for visibility_influence in range(1,5):
  #         try:
  #           pool.apply_async(influence_test, (graph, starts, ends, width, mean_costs, ant_count, ph_evap_coef, ph_influence, weight_influence, visibility_influence))
  #         except ValueError:
  #           pass
  #         # t = multiprocessing.Process(target=influence_test, args=(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence))
  #         # t.start()
  #         # threads.append(t)
                  
  #         # influence_test(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence)
  
  size = 50
  offset = 00
  N = 500
  test_range = arange(0.05,0.15,0.01)
  
  costs = manager.dict()
  times = manager.dict()
  sum_dict = {0: {"cost": 0, "time": 0}}
  allgen_dict = {0: {0: {"cost": 0, "time": 0}}}
  params = {'size': size, "Graphs": N, "offset": offset, "range": [test_range[0],test_range[-1]]}

  for ngen in test_range:
    sum_dict[ngen] = {"cost": 0, "time": 0}
    allgen_dict[ngen] = {0: {"cost": 0, "time": 0}}
    
  pool = multiprocessing.Pool(8)
  
  test_name = "ph_evap"
  
  for gnum in range(offset, offset + N):
    graph = ig.load(f"graphs/simulations/{size}x{size}/graph_{gnum}.graphml")
    start = int(graph["start"])
    end = int(graph["end"])
    
    if test_name == "generations":
      for ngen in test_range:
        pool.apply_async(generation_test, (graph, start, end, ngen, costs, times, gnum))
    elif  test_name == "ant_count":
      for ant_count in test_range:
        pool.apply_async(count_test, (graph, start, end, ant_count, costs, times, gnum))
    elif test_name == "ph_evap":
      for ph_evap_coef in test_range:
        pool.apply_async(evap_test, (graph, start, end, ph_evap_coef, costs, times, gnum))
      
  pool.close()
  pool.join()
  for i in range(offset, offset + N):
    for value in test_range:
      allgen_dict[value][i] = {"cost": costs[f"{i}:{value}"], "time": times[f"{i}:{value}"]}
      sum_dict[value]['cost'] += costs[f"{i}:{value}"]
      sum_dict[value]['time'] += times[f"{i}:{value}"]
  
  print(sum_dict)        
  
  jdict = {"params": params, "sum_dict" : sum_dict, "allgen_dict": allgen_dict}

  # print(mean_costs)
  fname = "dumps/" + test_name + "-" + str(datetime.now().strftime('%d-%m--%H-%M-%S')) + ".json"
  with open(fname, 'w') as f:
    dump(dict(jdict), f, indent=2)