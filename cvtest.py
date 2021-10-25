# %%
from random import randint
import cv2
import numpy as np
import igraph as ig
from math import floor

def generate_edge_list(width, height):
    edges = []
    for h in range(height - 1):
      for w in range(width - 1):
        vid = h * width + w
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
  for e in graph.es():
    e["weight"] = 1

  return graph

# %%

width = 120
height = 60
step = 10

graph = generate_graph(width, height)

# %%
img = np.ones((height*step,width*step,1), np.uint8)

path = graph.get_shortest_paths(1,width*height-100)

for v in path[0]:
  w = (v % width) * step
  h = floor(v/width) * step
  img[h:h+step,w:w+step] = 255

cv2.imshow('image',img)
cv2.waitKey(0)

# %%

while False:
  width *= step
  height *= step
  frame = np.ones((height,width,1), np.uint8)
  for h in range(0, height, step):
    for w in range(0, width, step):
      frame[h:h+step,w:w+step] = randint(0,255)

  # Display the resulting frame
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
      break

# When everything done, release the capture
cv2.destroyAllWindows()