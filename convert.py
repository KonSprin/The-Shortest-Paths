# %%
import os 
from igraph import *

file_name = "p2p-Gnutella08.txt"
# file_name = "web-Google.txt"

file_name_absolute = os.getcwd() + "\graphs\\" + file_name

def load_graph(file):
  edges = []

  lines = file.readlines()

  nnodes = int(lines[2].split(" ")[2])
  for line in lines[4:]:
    edges.append([int(s.strip()) for s in line.split("\t")])
  return nnodes, edges

#  %%
try:
  with open(file_name, "r") as f:
    nnodes, edges = load_graph(f)
except: 
  try: 
    with open(file_name_absolute, "r") as f:
      nnodes, edges = load_graph(f)
  except:
    pass

print(edges[-4:])

g = Graph(n=nnodes, edges=edges, directed=True)
g.save(file_name.split(".")[0] + ".graphml")