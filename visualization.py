# %%
import numpy as np
import logging as log
from spmodule.graphVis import *
from spmodule.splib import *

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')

# %%

width = 60
height = 40
step = 16
frame_width = width * step
frame_height = height * step

graph = generate_graph(width, height)

# %%
img = np.zeros((frame_height,frame_width,3), np.uint32)

# for w,h in draw_line(10,2,30,33):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255

# for w,h in draw_line(40,39, 50, 30):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255
  
# for w,h in draw_line(25,20, 10, 35):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255

# %%

start = wh2vid(3,20, width)
end =  wh2vid(45,30, width)
update_frame(width, step, end, img, 'b')

# random_walls(graph, img, step, 40, start, end)

path = graph.get_shortest_paths(1,100)
print(path)


# for v in graph.vs():
#   v["distance"] = diag_dist(v.index,end,width)
# graph.vs[end]["distance"] = 0.000001

# for e in graph.es():
#   e["weight"] = diag_dist(graph.vs[e.target].index,end,width)

for e in graph.es():
  e["weight"] = 1

# if True:
#   ig.save(graph, "graphs/basic.graphml")

astar_visualization(width, step, graph, img, start, end)

# greedy_visualization(width, step, graph, img, start, end)

# number_of_ants = 100
# ph_influence = 1
# weight_influence = 3
# ph_evap_coef=0.01
# ph_deposition=8000

# ant_visualization(width, step, graph, img, start, end, number_of_ants, ph_influence, weight_influence, ph_evap_coef, ph_deposition)