from os import close
import cv2
from spmodule.splib import *
import numpy as np

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
  fscore[start] = diag_dist(start, end, width) * (1+p)

  image_list = []
  while len(Q) > 0:
    tmp_dist = fscore.copy()
    for q in closed:
      tmp_dist[q] = float('inf')
    
    # tmp_dist = []
    # for q in range(vn):
    #   if q in Q:
    #     tmp_dist.append(fscore[q])
    #   else:
    #     tmp_dist.append(float('inf'))
    u = tmp_dist.index(min(tmp_dist))

    if u == end:
      path = reconstruct_path(start, u, previus)
      frame = img.copy()
      for v in Q:
        update_frame(width, step, v, frame, 'g')
      for v in closed:
        update_frame(width, step, v, frame, 'b')
      for v in path:
        update_frame(width, step, v, frame, 'w')
      for aaa in range(30):
        image_list.append(np.uint8(frame))
      
      return path, image_list

    Q.remove(u)
    closed.append(u)
    for v in graph.neighbors(u):
      if v in closed:
        continue
      
      eid = graph.get_eid(u, v)
      alt = dist[u] + graph.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        fscore[v] = alt + diag_dist(v, end, width) * (1+p)
    # fscore[v] = alt
        previus[v] = u
        if v not in Q:
            Q.append(v)
    
    frame = img

    for v in Q:
      update_frame(width, step, v, frame, 'g')

    for v in closed:
      update_frame(width, step, v, frame, 'b')

    image_list.append(np.uint8(frame))

  # Display the resulting frame
    cv2.imshow('frame', np.uint8(frame))
    if cv2.waitKey(1) == ord('q'):
      break

  # When everything done, release the capture
  cv2.destroyAllWindows()

def greedy_visualization(width, step, graph, img, start, end):
  vn = graph.vcount()

  previus = [[]] * vn

  opened_list = [start]
  closed = []

  dist = [float("inf")] * vn
  dist[start] = 0

  image_list = []
  while len(opened_list) > 0:
    tmp_dist = []
    for q in range(vn):
      if q in opened_list:
        tmp_dist.append(dist[q])
      else:
        tmp_dist.append(float('inf'))
    u = tmp_dist.index(min(tmp_dist))

    if u == end:
      path = reconstruct_path(start, u, previus)
      frame = img.copy()
      for v in opened_list:
        update_frame(width, step, v, frame, 'g')
      for v in closed:
        update_frame(width, step, v, frame, 'b')
      for v in path:
        update_frame(width, step, v, frame, 'w')
      for aaa in range(30):
        image_list.append(np.uint8(frame))
      
      return path, image_list

    opened_list.remove(u)
    closed.append(u)
    for v in graph.neighbors(u):
      alt = diag_dist(v, end, width)
      if alt < dist[v]:
        dist[v] = alt
        previus[v] = u
        if (v not in opened_list) and (v not in closed):
            opened_list.append(v)
    
    frame = img.copy()

    for v in opened_list:
      update_frame(width, step, v, frame, 'g')

    for v in closed:
      update_frame(width, step, v, frame, 'b')

    image_list.append(np.uint8(frame))
  # Display the resulting frame
    cv2.imshow('frame', np.uint8(frame))
    if cv2.waitKey(1) == ord('q'):
      break

  # When everything done, release the capture
  cv2.destroyAllWindows()

def dijkstra_visualization(g, start, end, img, step):
  vn = g.vcount()
  dist = [float("inf")] * vn
  previus = [[]] * vn

  opened = list(range(vn))
  
  dist[start] = 0
  closed = []

  width = g["width"]
  
  image_list = []
  
  while len(opened) > 0:
    tmp_dist = dist.copy()
    for q in closed:
      tmp_dist[q] = float('inf')
      
    u = tmp_dist.index(min(tmp_dist))
    opened.remove(u)
    closed.append(u)
    
    if u == end: 
      path = reconstruct_path(start, end, previus)
      
      frame = img.copy()
      for v in closed:
        update_frame(width, step, v, frame, 'b')
      for v in path:
        update_frame(width, step, v, frame, 'g')
      for aaa in range(30):
        image_list.append(np.uint8(frame))
      
      return path, image_list
    
    for v in g.neighbors(u):
      eid = g.get_eid(u, v)
      alt = dist[u] + g.es(eid)["weight"][0]
      if alt < dist[v]:
        dist[v] = alt
        previus[v] = u

    frame = img.copy()
    
    # for v in opened:
    #   update_frame(width, step, v, frame, 'g')

    for v in closed:
      update_frame(width, step, v, frame, 'b')

    image_list.append(np.uint8(frame))

  # Display the resulting frame
    cv2.imshow('frame', np.uint8(frame))
    if cv2.waitKey(1) == ord('q'):
      break

  # When everything done, release the capture
  cv2.destroyAllWindows()
      
def ant_visualization(width, step, graph, img, start, end, number_of_ants, ph_influence, weight_influence, ph_evap_coef, visibility_influence):
  for v in graph.vs():
    v["distance"] = diag_dist(v.index,end,width)
  graph.vs[end]["distance"] = 0.1

  # def update_frame(width, step, v, frame, value):
  #   w, h = [x * step for x in vid2wh(v, width)]
  #   frame[h:h+step,w:w+step,1] -= value

  ph_deposition = graph.ecount() * ph_evap_coef
  
  best_gen_path = []
  best_gen_path_weight = []
  num_reached = 0
  
  image_list = []
  while len(image_list) < 150:
    all_paths, all_paths_weight = ant_edge_selection(graph, start, end, number_of_ants, ph_influence, weight_influence, visibility_influence)
    # graph = pheromone_update(graph, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)
    
      
    if len(all_paths) > 0:
      best_gen_path_index = all_paths_weight.index(min(all_paths_weight))
      best_gen_path.append(all_paths[best_gen_path_index])
      best_gen_path_weight.append(all_paths_weight[best_gen_path_index])
      log.debug("Best path this generation: " + str(best_gen_path_weight[-1]))

      num_reached += len(all_paths)
      
    lprint(f"Ended generation with best paths cost: {best_gen_path_weight[-1]}")    
    ## Pheromone update after each generation
    # graph = pheromone_update(graph, ph_evap_coef, ph_deposition, all_paths, all_paths_weight)
    graph = minmax_pheromone_update(graph, ph_evap_coef, ph_deposition*100, best_gen_path[-1], best_gen_path_weight[-1])

    # for path in all_paths:
    #   for v in path:
    #     update_frame(width, step, v, frame, 10)

    # tmp_frame = np.zeros((frame_height,frame_width,1), np.uint8)
    frame = img.copy()

    # for e in graph.es():
    #   s = e.source
    #   t = e.target
    #   update_frame(width, step, s, frame, np.uint8(e["pheromone"]))
    #   update_frame(width, step, t, frame, np.uint8(e["pheromone"]))
    # frmax = frame[:,:,1].max()
    # print(frmax)
    # frame[:,:,1] =  (frame[:,:,1]/frmax) * 254
    
    for v in best_gen_path[-1]:
      update_frame(width, step, v, frame, 'b')
    image_list.append(np.uint8(frame))
    cv2.imshow('frame', np.uint8(frame))
    if cv2.waitKey(1) == ord('q'):
      break
  
  return image_list
      


  # w, h = [x * step for x in vid2wh(s, width)]
  # tmp_frame[h:h+step,w:w+step] += e["pheromone"]
  # w, h = [x * step for x in vid2wh(t, width)]
  # tmp_frame[h:h+step,w:w+step] += e["pheromone"]
  # max = tmp_frame.max()

  # print(all_paths)

  # Display the resulting frame
    # cv2.imshow('frame', np.uint8(frame))
    # if cv2.waitKey(1) == ord('q'):
    #   break

  # When everything done, release the capture
