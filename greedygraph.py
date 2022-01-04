import numpy as np
import logging as log
from spmodule.graphVis import *
from spmodule.splib import *


if False:
  width = 9
  height = 6
  step = 100

  start = wh2vid(0,3, width)
  end =  wh2vid(8,3, width)

  frame_width = width * step
  frame_height = height * step
  graph = generate_graph(width, height)
  img = np.ones((frame_height,frame_width,3), np.uint32) * 255

  for w,h in draw_line(2,3,7,3):
    graph.delete_edges(graph.incident(wh2vid(w,h,width)))
    img[h*step:h*step+step,w*step:w*step+step] = 0

  for w,h in draw_line(3,1,3,2):
    graph.delete_edges(graph.incident(wh2vid(w,h,width)))
    img[h*step:h*step+step,w*step:w*step+step] = 0
    
  for w,h in draw_line(5,0,5,1):
    graph.delete_edges(graph.incident(wh2vid(w,h,width)))
    img[h*step:h*step+step,w*step:w*step+step] = 0
    
  for w,h in draw_line(7,1,7,2):
    graph.delete_edges(graph.incident(wh2vid(w,h,width)))
    img[h*step:h*step+step,w*step:w*step+step] = 0

  for w,h in draw_line(2,3,2,4):
    graph.delete_edges(graph.incident(wh2vid(w,h,width)))
    img[h*step:h*step+step,w*step:w*step+step] = 0


  path = bestfirst(graph, start, end)

  for v in path:
    update_frame(width, step, v ,img, 'g')
  update_frame(width, step, start ,img, 'b')
  update_frame(width, step, end ,img, 'r')

  cv2.imshow('frame', np.uint8(img))
  cv2.imwrite(f"visualizations/greedy.png", np.uint8(img))
  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
else:
  width = 30
  height = 20
  step = 50

  start = wh2vid(0,10, width)
  end =  wh2vid(29,10, width)

  frame_width = width * step
  frame_height = height * step
  graph = generate_graph(width, height)
  img = np.ones((frame_height,frame_width,3), np.uint32) * 255
  
  width = graph["width"]
  for v in graph.vs():
    v["distance"] = diag_dist(v.index, end, width)
  graph.vs[end]["distance"] = 0.0001
  
  def rand_path(graph, start, stop):
    while True:
      current = start
      visited = []
      while current != stop:
        possible = setsub(graph.neighbors(current), visited)
        
        if len(possible) == 0: break
        
        dist = []
        for way in possible:
          dist.append(1/graph.vs[way]["distance"])
          
        dist = [d - min(dist)/1.2 for d in dist]
        visited.append(current)
        current = choices(possible, dist)[0]
      
      if current == stop: return visited
        
  path = rand_path(graph, start, end)
  
  for v in path:
    update_frame(width, step, v ,img, 'g')
  update_frame(width, step, start ,img, 'b')
  update_frame(width, step, end ,img, 'r')
      
  cv2.imshow('frame', np.uint8(img))
  cv2.imwrite(f"visualizations/random_path1.png", np.uint8(img))
  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
    
  img = np.ones((frame_height,frame_width,3), np.uint32) * 255
  for v in remove_loops(graph,path):
    update_frame(width, step, v ,img, 'g')
  update_frame(width, step, start ,img, 'b')
  update_frame(width, step, end ,img, 'r')
      
  cv2.imshow('frame', np.uint8(img))
  cv2.imwrite(f"visualizations/random_path_removed1.png", np.uint8(img))
  if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()