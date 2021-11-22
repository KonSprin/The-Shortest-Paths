# %%

import multiprocessing
from multiprocessing import managers
import igraph as ig 
import logging as log
from spmodule.splib import *
from json import dump
from datetime import datetime
import multiprocessing
from time import sleep
from random import randint

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
    
if __name__ == '__main__':

  log.basicConfig(level=log.INFO,
                  format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                  filename='sp.log', filemode='w')

  # multiprocessing.set_start_method("spawn")

  LOG = log.getLogger()

  threads = []
  pool = multiprocessing.Pool(8)

  graphs = ["graphs/optimization100x50.graphml", "graphs/optimization60x40.graphml", "graphs/optimization30x30.graphml"]
  starts_dict = {"graphs/optimization100x50.graphml": [0, 4508, 0, 4508, 0, 350, 350, 3501], 
                "graphs/optimization60x40.graphml": [0, 59, 0, 59, 2399, 2301], 
                "graphs/optimization30x30.graphml": [0, 29, 0, 29, 899, 871]}
  ends_dict = {"graphs/optimization100x50.graphml": [4999, 99, 99, 4999, 4508, 4999, 4508, 3599], 
              "graphs/optimization60x40.graphml": [2399, 2301, 59, 2399, 2301, 0], 
              "graphs/optimization30x30.graphml": [899, 871, 29, 899, 871, 0]}

  test_name = "influences"

  # class myThread (threading.Thread):
  #    def __init__(self, graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence):
  #       threading.Thread.__init__(self)
  #       self.graph = graph
  #       self.starts = starts
  #       self.ends = ends
  #       self.width = width
  #       self.mean_costs = mean_costs
  #       self.ph_influence = ph_influence
  #       self.weight_influence = weight_influence
  #       self.visibility_influence = visibility_influence
  #    def run(self):
  #       influence_test(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence)

  manager = multiprocessing.Manager()
  mean_costs = manager.dict()
  
  for graph_name in graphs:
    graph = ig.load(graph_name)

    starts = starts_dict[graph_name]
    ends =   ends_dict[graph_name]
    
    width = graph["width"]

    if test_name == "ph_evap":
      
      for ph_evap_coef in range(1,20,1):
        ph_evap_coef = ph_evap_coef/100
        cost_sum = 0
        for start, end in zip(starts,ends):
          path = antss(graph, start, end, 10, 20, ph_evap_coef)
          cost_sum += path_cost(graph, path)
        
        mean_costs[ph_evap_coef] = cost_sum
        print(cost_sum)
    
    elif test_name == "influences":
      for ph_influence in range(1,5):
        for weight_influence in range(1,5):
          for visibility_influence in range(1,5):
            # t = multiprocessing.Process(target=influence_test, args=(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence))
            # t.start()
            # threads.append(t)
            
            # threads.append((graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence))
            try:
              pool.apply_async(influence_test, (graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence))
            except ValueError:
              pass
            # influence_test(graph, starts, ends, width, mean_costs, ph_influence, weight_influence, visibility_influence)
            
  pool.close()
  pool.join()
  # for th in threads:
  #   th.join()

  
  print(mean_costs)
  fname = "dumps/" + test_name + "-" + str(datetime.now().strftime('%d-%m--%H-%M-%S')) + ".json"
  with open(fname, 'w') as f:
    dump(dict(mean_costs), f, indent=2)