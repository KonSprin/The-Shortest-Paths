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
  width = 100
  height = 100
  step = 10

  start = wh2vid(5,5, width)
  end =  wh2vid(95,95, width)

  no_mountains = 6
  mountain_height = 15
  wall_percent = 15
  graph, img = generate_weighted_graph(width, height, step, start, end, mountain_height, wall_percent)

else: 
  size = 100
  step = 10
  start = wh2vid(0,0, size)
  end =  wh2vid(29,19, size)
  
  graph = ig.load(f"graphs/simulations/{size}x{size}/graph_{1}.graphml")
  start = int(graph["start"])
  end = int(graph["end"])
    
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

# print(path_cost(graph, Astar(graph,start,end)))

# number_of_ants = 20
# ph_evap_coef=0.05
# ph_influence = 1
# weight_influence = 8
# visibility_influence  = 1

# image_list = ant_visualization(width, step, graph, img, start, end, 
#                   number_of_ants, ph_influence, weight_influence,
#                   ph_evap_coef, visibility_influence)

# print(len(image_list))
# imageio.mimsave('graphs/a_star2.gif', image_list, fps=30)
  
djpath = dijkstra(graph, start, end)
djimg = img.copy()
for v in djpath:
   update_frame(width, step, v , djimg, 'b')
  
aspath = Astar(graph, start, end)
asimg = img.copy()
for v in aspath:
   update_frame(width, step, v , asimg, 'b')
  
apath = antss(graph, start, end)
aimg = img.copy()
for v in apath:
   update_frame(width, step, v , aimg, 'b')
  
bspath = bestfirst(graph, start, end)
bsimg = img.copy()
for v in bspath:
   update_frame(width, step, v , bsimg, 'b')
  
  
cv2.imshow('frame', np.uint8(img))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()
  
cv2.imshow('frame', np.uint8(djimg))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()
  
cv2.imshow('frame', np.uint8(asimg))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()
  
cv2.imshow('frame', np.uint8(aimg))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()

cv2.imshow('frame', np.uint8(bsimg))
if cv2.waitKey(0) == ord('q'):
  cv2.destroyAllWindows()
  
  
cv2.imwrite(f"visualizations/no_path.png", np.uint8(img))
cv2.imwrite(f"visualizations/dijkstra_path.png", np.uint8(djimg))
cv2.imwrite(f"visualizations/astar_path.png", np.uint8(asimg))
cv2.imwrite(f"visualizations/ants_path.png", np.uint8(aimg))
cv2.imwrite(f"visualizations/bestfirst_path.png", np.uint8(bsimg))

  # %%
