# %%
import numpy as np
import logging as log
from spmodule.graphVis import *
from spmodule.splib import *
from random import sample

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')
log.captureWarnings(True)
# import warnings
# warnings.filterwarnings("error")

# %%

width = 60
height = 40
step = 16
frame_width = width * step
frame_height = height * step

start = wh2vid(3,20, width)
end =  wh2vid(45,30, width)

path = [[]]
while path == [[]]:
  graph = generate_graph(width, height)
  img = np.zeros((frame_height,frame_width,3), np.uint32)
  
  add_mountains(graph, img, sample(range(width*height), 6), 10, step)
  random_points(graph, img, step, 20, start, end)
  try: 
    path = graph.get_shortest_paths(start, end)
  except RuntimeWarning: 
    log.warning("lmao")
    pass
print(path)

update_frame(width, step, end, img, 'b')
update_frame(width, step, start, img, 'b')

# for w,h in draw_line(10,2,30,33):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255

# for w,h in draw_line(40,39, 50, 30):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255
  
# for w,h in draw_line(25,20, 10, 35):
#   graph.delete_edges(graph.incident(wh2vid(w,h,width)))
#   img[h*step:h*step+step,w*step:w*step+step,2] = 255

# for v in graph.vs():
#   v["distance"] = diag_dist(v.index,end,width)
# graph.vs[end]["distance"] = 0.000001

# for e in graph.es():
#   e["weight"] = diag_dist(graph.vs[e.target].index,end,width)

for e in graph.es():
  e["weight"] = 1

# if True:
#   ig.save(graph, "graphs/basic.graphml")

path = astar_visualization(width, step, graph, img, start, end)

# path = greedy_visualization(width, step, graph, img, start, end)

# number_of_ants = 100
# ph_influence = 1
# weight_influence = 3
# ph_evap_coef=0.01
# ph_deposition=8000

# ant_visualization(width, step, graph, img, start, end, number_of_ants, ph_influence, weight_influence, ph_evap_coef, ph_deposition)

for v in path:
  update_frame(width, step, v ,img, 'w')
  
cv2.imshow('frame', np.uint8(img))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()