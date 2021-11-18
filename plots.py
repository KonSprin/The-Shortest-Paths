from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import json

with open('data/stat.json') as json_file:
  data = json.load(json_file)
  
dist_ig = data[0]
dist_gr = data[1]
dist_bf = data[2]
dist_dj = data[3]
dist_as = data[4]

times_ig = data[5]
times_gr = data[6]
times_bf = data[7]
times_dj = data[8]
times_as = data[9]

# fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.

plt.plot(list(range(len(dist_ig))), dist_ig, 
         list(range(len(dist_gr))), dist_gr, 
         list(range(len(dist_bf))), dist_bf, 
         list(range(len(dist_dj))), dist_dj, 
         list(range(len(dist_as))), dist_as)  # Matplotlib plot.

plt.show()

plt.plot(list(range(len(dist_ig))), times_ig, 
         list(range(len(dist_gr))), times_gr, 
         list(range(len(dist_bf))), times_bf, 
         list(range(len(dist_dj))), times_dj, 
         list(range(len(dist_as))), times_as) 

plt.show()