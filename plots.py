from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import json

def merge_files(in1, in2, outname):
  with open(in1) as json_file:
    data = json.load(json_file)

  params = data["params"]
  sum_dict = data["sum_dict"]

  with open(in2) as json_file:
    data2 = json.load(json_file)

  r = params["range"]
  test_range = np.arange(r[0], r[1], r[2], dtype=np.longdouble)

  for i in range(5,51):
    # i = i/100
    for key in sum_dict["0"].keys():
      sum_dict[str(i)][key] += data2["sum_dict"][str(i)][key]

  params["Graphs"] += data2["params"]["Graphs"]
  
  outdata = {"params": params, "sum_dict": sum_dict}
  
  # fname = f"dumps/data/{outname}{params['size']}x1000.json"
  with open(outname, 'w') as f:
    json.dump(outdata, f, indent=2)

def dumps2data(inname, outname):
  with open(inname) as json_file:
    data = json.load(json_file)
  
  sum_dict = data["sum_dict"]
  
  with open(outname, 'w') as f:
    json.dump(sum_dict, f, indent=2)
    
# fname = "dumps/data/antcount_test20x200.json"
# dumps2data("dumps/ant_count-05-12--23-45-57.json", fname)
merge_files("dumps/final/astar_hk-02-01--23-28-22.json", "dumps/final/astar_hk-03-01--00-04-29.json", "dumps/final/astar_hk_merged.json")

with open("dumps/final/astar_hk_merged.json") as json_file:
  data = json.load(json_file)
  
# with open("dumps/AS/ph_evap-29-12--09-04-59.json") as json_file:
#   data2 = json.load(json_file)
  
params = data["params"]
r = params["range"]
gnum = params["Graphs"]

# test_range = np.arange(0.05, 0.19, 0.01)
test_range = range(5,51)
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # Plot some data on the axes.
costs = []
times = []
cvt = []
for i in test_range:
  # i = i/100
  val = data["sum_dict"][str(i)]
  costs.append(val["cost"]/gnum)
  times.append(val["time"]/gnum)
  cvt.append(val["cost"]/val["time"])
  
# for i in range(25,35):
#   # i = i/100
#   val = data2["sum_dict"][str(i)]
#   costs.append(val["cost"]/gnum)
#   times.append(val["time"]/gnum)
#   cvt.append(val["cost"]/val["time"])

print(costs)
print(times)
t = [xx/5 for xx in test_range]
print(t)
if False:
  fig, ax = plt.subplots()
  ax.set_title("Kosz/czas dla grafu 20x20")
  ax.set_xlabel('liczba mrówek')
  ax.set_ylabel('koszt/czas')
  ax.plot(t, cvt)
else:
  fig, ax1 = plt.subplots()
  color = 'tab:red'
  ax1.set_title("Zależność współczynnika heurystyki dla grafu 100x100")
  ax1.set_xlabel('współczynnik')
  ax1.set_ylabel('średni koszt', color=color)
  ax1.plot(t, costs, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('średni czas [s]', color=color)  # we already handled the x-label with ax1
  ax2.plot(t, times, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.show()
