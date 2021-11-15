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

# %%

width = 60
height = 40
step = 16
frame_width = width * step
frame_height = height * step

graph = generate_graph(width, height)

# %%
img = np.zeros((frame_height,frame_width,3), np.uint32)

for w,h in draw_line(10,2,30,33):
  graph.delete_edges(graph.incident(wh2vid(w,h,width)))
  img[h*step:h*step+step,w*step:w*step+step,2] = 255

for w,h in draw_line(40,39, 50, 30):
  graph.delete_edges(graph.incident(wh2vid(w,h,width)))
  img[h*step:h*step+step,w*step:w*step+step,2] = 255


path = graph.get_shortest_paths(1,100)
print(path)

# %%

start = wh2vid(3,20, width)
end =  wh2vid(45,30, width)
update_frame(width, step, end, img, 'r')

# for v in graph.vs():
#   v["distance"] = diag_dist(v.index,end,width)
# graph.vs[end]["distance"] = 0.000001

# for e in graph.es():
#   e["weight"] = diag_dist(graph.vs[e.target].index,end,width)

for e in graph.es():
  e["weight"] = 1

# if True:
#   ig.save(graph, "graphs/basic.graphml")

def astar_visualization(width, step, graph, img, start, end):
  vn = graph.vcount()
  p = 2/vn

  previus = [[]] * vn

  Q = [start]
  closed = []

  dist = [float("inf")] * vn
  dist[start] = 0

  fscore = [float("inf")] * vn
  # fscore[start] = 0
  fscore[start] = diag_dist(start, end, graph["width"]) * (1+p)

  while len(Q) > 0:
    tmp_dist = []
    for q in range(vn):
      if q in Q:
        tmp_dist.append(fscore[q])
      else:
        tmp_dist.append(float('inf'))
    u = tmp_dist.index(min(tmp_dist))

    if u == end:
      path = reconstruct_path(start, u, previus)
      for v in path:
        update_frame(width, step, v, frame, 'w')
      if cv2.waitKey(0) == ord('q'):
        break

    Q.remove(u)
    closed.append(u)
    for v in graph.neighbors(u):
      eid = graph.get_eid(u, v)
      alt = dist[u] + graph.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        fscore[v] = alt + diag_dist(v, end, graph["width"]) * (1+p)
    # fscore[v] = alt
        previus[v] = u
        if v not in Q:
            Q.append(v)
    
    frame = img

    for v in Q:
      update_frame(width, step, v, frame, 'g')

    for v in closed:
      update_frame(width, step, v, frame, 'b')


  # Display the resulting frame
    cv2.imshow('frame', np.uint8(frame))
    if cv2.waitKey(1) == ord('q'):
      break

  # When everything done, release the capture
  cv2.destroyAllWindows()

astar_visualization(width, step, graph, img, start, end)