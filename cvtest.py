# %%
from random import randint
import cv2
import numpy as np
import igraph as ig
from math import floor
from splib import *
import logging as log

log.basicConfig(level=log.DEBUG,
                filename='sp.log', filemode='w', 
                format='%(levelname)s: %(module)s %(asctime)s %(message)s')

def generate_edge_list(width, height):
    edges = []
    for h in range(height - 1):
      for w in range(width - 1):
        vid = wh2vid(width, w, h)
        edges.append((vid,vid + 1))
        edges.append((vid,vid + width))
      vid += 1
      edges.append((vid, vid + width))
    for w in range(width-1):
      vid = width*height - width + w
      edges.append((vid, vid + 1))
    return edges
    
def generate_graph(width, height):
  vnum = int(width*height)
  graph = ig.Graph(vnum)

  edges = generate_edge_list(width, height)
  graph.add_edges(edges)

  return graph

def vid2wh(v, width):
  w = (v % width)
  h = floor(v/width)
  return w,h

def wh2vid(width, w, h):
    vid = h * width + w
    return vid
# %%

width = 120
height = 60
step = 10
frame_width = width * step
frame_height = height * step

graph = generate_graph(width, height)

# %%
img = np.ones((frame_height,frame_width,1), np.uint8)*255

path = graph.get_shortest_paths(1,100)
print(path)

# for v in path[0]:
#   w, h = [x * step for x in vid2wh(v, width)]
#   img[h:h+step,w:w+step] = 0

# cv2.imshow('image',img)
# cv2.waitKey(0)

# %%
# Initial value of pheromone on each edge 
for e in graph.es():
  e["pheromone"] = 1

start = 0
end = 120*20 + 20
number_of_ants = 100
ph_influence = 1
weight_influence = 1
ph_evap_coef=0.01
ph_deposition=5

for e in graph.es():
  e["weight"] = np.linalg.norm(np.array(vid2wh(e.target, width)) - np.array(vid2wh(end, width)))

while True:
  frame = np.ones((frame_height,frame_width,1), np.uint8) * 255 
  # for h in range(0, height, step):
  #   for w in range(0, width, step):
  #     frame[h:h+step,w:w+step] = randint(0,255)

  all_paths, all_paths_weight = ant_edge_selection(graph, start, end, number_of_ants, ph_influence, weight_influence)
  graph = pheromone_update(graph, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)

  for path in all_paths:
    for v in path:
      w, h = [x * step for x in vid2wh(v, width)]
      frame[h:h+step,w:w+step] -= 10

  print(all_paths)

  # Display the resulting frame
  cv2.imshow('frame', frame)
  if cv2.waitKey(25) == ord('q'):
      break

# When everything done, release the capture
cv2.destroyAllWindows()