# %%

import igraph as ig 
import logging as log
from spmodule.splib import *
from json import dump

log.basicConfig(level=log.INFO,
                format='%(levelname)s: %(module)s %(asctime)s %(message)s',
                filename='sp.log', filemode='w')


LOG = log.getLogger()


width = 100
height = 50
step = 10

graph = ig.load("graphs/optimization.graphml")


# %%

starts = [0   , 4508, 0 , 4508, 0   , 350 , 350 , 3501]
ends =   [4999, 99  , 99, 4999, 4508, 4999, 4508, 3599]

mean_costs = {}
for ph_evap_coef in range(1,20,1):
  ph_evap_coef = ph_evap_coef/100
  cost_sum = 0
  for start, end in zip(starts,ends):
    path = antss(graph, start, end, 10, 20, ph_evap_coef)
    cost_sum += path_cost(graph, path)
  
  mean_costs[ph_evap_coef] = cost_sum
  print(cost_sum)

print(mean_costs)
with open("dumps/ph_evap_coef_test3.json", 'w') as f:
  dump(mean_costs, f)