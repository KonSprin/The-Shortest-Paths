# %%
import numpy as np
import logging as log
from spmodule.graphVis import *
from spmodule.splib import *
import imageio

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')
log.captureWarnings(True)
# import warnings
# warnings.filterwarnings("error")

# %%
if True:
  width = 60
  height = 40
  step = 16

  start = wh2vid(0,0, width)
  end =  wh2vid(59,39, width)

  no_mountains = 6
  mountain_height = 15
  wall_percent = 15
  graph, img = generate_weighted_graph(width, height, step, start, end, no_mountains, mountain_height, wall_percent)

else: 
  width = 30
  height = 20
  step = 32
  start = wh2vid(0,0, width)
  end =  wh2vid(29,19, width)
  
  graph = ig.load("graphs/example.graphml")
  with open("graphs/example.npy", "rb") as f:
    img = np.load(f)
# ig.save(graph, "graphs/basic.graphml")


# %%

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


# if True:
#   ig.save(graph, "graphs/basic.graphml")

# print(sum([graph.vs(v)["height"][0] for v in dijkstra(graph, start, end)]))

# path, image_list = astar_visualization(width, step, graph, img, start, end)
# astart_cost = sum([graph.vs(v)["height"][0] for v in path])
# print(astart_cost)

# for v in path:
#   update_frame(width, step, v ,img, 'w')
  
# path, image_list = greedy_visualization(width, step, graph, img, start, end)
# greedy_cost = path_cost(graph, path)
# print(greedy_cost)

# for v in path:
#   update_frame(width, step, v ,img, 'w')

# path, image_list = dijkstra_visualization(graph, start, end, img, step)
# dijkstra_cost = path_cost(graph, path)
# print(dijkstra_cost)

# for v in path:
#   update_frame(width, step, v ,img, 'w')

print(path_cost(graph, Astar(graph,start,end)))

number_of_ants = 20
ph_evap_coef=0.05
ph_influence = 1
weight_influence = 8
visibility_influence  = 1

image_list = ant_visualization(width, step, graph, img, start, end, 
                  number_of_ants, ph_influence, weight_influence,
                  ph_evap_coef, visibility_influence)

# print(len(image_list))
# imageio.mimsave('graphs/a_star2.gif', image_list, fps=30)
  
cv2.imshow('frame', np.uint8(img))

if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()
# %%
