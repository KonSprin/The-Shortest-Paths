import multiprocessing
import igraph as ig
from spmodule.splib import *




def gg(size, width, height, mountain_height, wall_percent, i):
  timer = Timer()
  gname = f"graphs/simulations/{size}x{size}/graph_{i}.graphml"
  start, target = sample(range(width*height - 1), 2)

  graph = generate_weighted_graph_noimg(width, height, start, target, mountain_height, wall_percent)
  graph["start"] = start
  graph["end"] = target
  graph["mountain_height"] = mountain_height
  graph["wall_percent"] = wall_percent

  ig.save(graph, gname)

  print(f"generated {i} in {timer.time()}")

if __name__ == '__main__':
  size = 300

  width = size
  height = size 

  mountain_height = 10
  wall_percent = 5

  N = 1000
  
  pool = multiprocessing.Pool(8)
  
  for i in range(500):
    pool.apply_async(gg, (size, width, height, mountain_height, wall_percent, i))
    
    
  pool.close()
  pool.join()    
    
