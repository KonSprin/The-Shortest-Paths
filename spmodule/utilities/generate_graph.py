import igraph
from igraph.drawing.colors import color_name_to_rgb

try: 
  g = igraph.load("graph.graphml")
except: 
  print("A")
  g = igraph.Graph().Erdos_Renyi(n=3000,p=0.05)

# path = g.get_eids(path=g.get_shortest_paths(1,9)[0]) 
# edge_list = g.get_eids(g.get_edgelist())

visual_style = {}
# visual_style["edge_width"] = [1+2*int(x) for x in [e in path for e in edge_list]]
visual_style["vertex_label"] = [a for a in range(g.vcount())]
visual_style["layout"] = g.layout("kk")

colors = []
ver_to_delete = []
for id, degree in enumerate(g.degree()):
  if degree == 0:
    ver_to_delete.append(id)
    colors.append("red")
  else:
    colors.append("blue")

visual_style["vertex_color"] = colors
igraph.plot(g, **visual_style)

g.delete_vertices(ver_to_delete)

colors = []
for degree in g.degree():
  if degree == 0:
    colors.append("red")
  else:
    colors.append("blue")

visual_style["vertex_label"] = [a for a in range(g.vcount())]
visual_style["vertex_color"] = colors
igraph.plot(g, **visual_style)

igraph.save(g, "graph.graphml")
