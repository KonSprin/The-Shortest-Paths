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
      vid = wh2vid(w, h, width)
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

def wh2vid(w, h, width):
    vid = h * width + w
    return vid

def draw_line(x0, y0, x1, y1):
  line=[(x0, y0)]
  dx = np.abs(x1-x0)
  sx = 1 if x0 < x1 else -1
  dy = -np.abs(y1-y0)
  sy = 1 if y0 < y1 else -1
  err = dx + dy

  while not ((x0 == x1) & (y0 == y1)):
    e2 = err * 2

    if e2 > dy:
      err += dy
      x0 += sx
      line.append((x0,y0))
    elif e2 < dx:
      err += dx
      y0 += sy
      line.append((x0,y0))
    # print (x0,y0)
  return line
    
  

# %%

width = 60
height = 40
step = 16
frame_width = width * step
frame_height = height * step

graph = generate_graph(width, height)

# %%
img = np.zeros((frame_height,frame_width,1), np.uint32)

for w,h in draw_line(30,2,30,33):
  graph.delete_edges(graph.incident(wh2vid(w,h,width)))
  img[h*step:h*step+step,w*step:w*step+step] = 255


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

start = wh2vid(3,20, width)
end =  wh2vid(55,30, width)
number_of_ants = 100
ph_influence = 1
weight_influence = 3
ph_evap_coef=0.01
ph_deposition=8000

for v in graph.vs():
  v["distance"] = np.linalg.norm(np.array(vid2wh(v.index, width)) - np.array(vid2wh(end, width)))
graph.vs[end]["distance"] = 0.1

def update_frame(width, step, v, frame, value):
  w, h = [x * step for x in vid2wh(v, width)]
  frame[h:h+step,w:w+step] += value

while True:
  # for h in range(0, height, step):
  #   for w in range(0, width, step):
  #     frame[h:h+step,w:w+step] = randint(0,255)

  all_paths, all_paths_weight = ant_edge_selection(graph, start, end, number_of_ants, ph_influence, weight_influence)
  graph = pheromone_update(graph, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)

  # for path in all_paths:
  #   for v in path:
  #     update_frame(width, step, v, frame, 10)

  # tmp_frame = np.zeros((frame_height,frame_width,1), np.uint8)
  frame = img

  for e in graph.es():
    s = e.source
    t = e.target
    update_frame(width, step, s, frame, np.uint8(e["pheromone"]))
    update_frame(width, step, t, frame, np.uint8(e["pheromone"]))

  print(frame.max())
  frame =  np.uint8((frame/frame.max()) * 254)

    # w, h = [x * step for x in vid2wh(s, width)]
    # tmp_frame[h:h+step,w:w+step] += e["pheromone"]
    # w, h = [x * step for x in vid2wh(t, width)]
    # tmp_frame[h:h+step,w:w+step] += e["pheromone"]
    # max = tmp_frame.max()

  # print(all_paths)

  # Display the resulting frame
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
      break

# When everything done, release the capture
cv2.destroyAllWindows()