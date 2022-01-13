import json
import csv
from igraph import write
import numpy as np


def merge_files(in1, in2, outname):
  with open(in1) as json_file:
    data1 = json.load(json_file)

  params = data1["params"]
  dict1 = data1["all_dict"]
  meantime1 = data1["meantime"]
  mean_cost1 = data1["mean_cost"]
  
  with open(in2) as json_file:
    data2 = json.load(json_file)
    
  dict2 = data2["all_dict"]
  meantime2 = data2["meantime"]
  mean_cost2 = data2["mean_cost"]
  
  all_dict = {}
  all_dict.update(dict1)
  all_dict.update(dict2)
  meantime = (meantime1 + meantime2)/2
  mean_cost = (mean_cost1 + mean_cost2)/2
  params["Graphs"] += data2["params"]["Graphs"]
  
  outdata = {"params": params, 
             "meantime": meantime, 
             "mean_cost": mean_cost, 
             "all_dict": all_dict}
  
  with open(outname, 'w') as f:
    json.dump(outdata, f, indent=2)

# merge_files("dumps/final/dijkstra-100--03-01--01-04-46.json", "dumps/final/dijkstra-100--03-01--00-58-18.json", "dumps/final/dijkstra-100-merged.json")

algo = "bestfirst"
filename = f"dumps/final/{algo}-100-merged.json"

with open(filename) as json_file:
  data = json.load(json_file)
  
dict = data["all_dict"]
number = []
cost = []
time = []

with open(filename.split(".")[0]+".csv", 'w', newline='') as csvfile:
  writer = csv.writer(csvfile, delimiter=' ',
                      quotechar='|', quoting=csv.QUOTE_MINIMAL)

  writer.writerow(["number", "cost", "time"])

  for index in dict:
    if index == "0": continue
    c = dict[index]["cost"]
    t = dict[index]["time"]
    
    writer.writerow([index, c, t])
    
    # print(f"{index}:{c}:{t}")
    # number.append(int(index))
    # cost.append(c)
    # time.append(t)

