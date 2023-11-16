import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_util(test, label):
  threads = []
  times = []
  for ms in test['data']:
    threads.append(ms['threads'])
    times.append(ms['time'])

  plt.plot(times, marker='o', label=f"{test[label.lower()]}")

def plot_time(tests, filename, legend, title):
  fig, ax = plt.subplots(figsize=(10, 6))

  for test in tests:
    plot_util(test, legend)

  plt.xlabel('Threads')
  plt.ylabel('Time')
  plt.suptitle(title)
  plt.title("Clusters " + str(test['clusters']) + " Bindstrat :" + str(test['bindstrat']))
  plt.legend(title = legend)
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  
  
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def plot_util_seq(test, label):
  threads = []
  times = []
  seqtime = test['seqtime']
  for ms in test['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      times.append(None)
    else:
      times.append(seqtime / ms['time'])

  plt.plot(times, marker='o', label=f"{test[label.lower()]}")

def plot_seqtime(tests, filename, legend, title):
  fig, ax = plt.subplots(figsize=(10, 6))

  for test in tests:
    plot_util_seq(test, legend)

  plt.xlabel('Threads')
  plt.ylabel('Speedup')
  plt.suptitle(title)
  plt.title("Clusters " + str(test['clusters']) + " Bindstrat :" + str(test['bindstrat']))
  plt.legend(title = legend)
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

with open(sys.argv[1]) as f:
  data = json.load(f)

plot_time(filter(lambda x: x['clusters'] == 16 and x['bindstrat'] == 1, data), "compare_cr_16_1_time.png", "Implementation", "Critical vs Reduction")
plot_time(filter(lambda x: x['clusters'] == 16 and x['bindstrat'] == 2, data), "compare_cr_16_2_time.png", "Implementation", "Critical vs Reduction")
plot_time(filter(lambda x: x['implementation'] == 'reduction' and x['bindstrat'] == 1, data), "compare_redu_1_time.png", "Clusters", "Clusters comparison")

plot_seqtime(filter(lambda x: x['clusters'] == 16 and x['bindstrat'] == 1, data), "compare_cr_16_1_seqtime.png", "Implementation", "Critical vs Reduction")
plot_seqtime(filter(lambda x: x['clusters'] == 16 and x['bindstrat'] == 2, data), "compare_cr_16_2_seqtime.png", "Implementation", "Critical vs Reduction")
plot_seqtime(filter(lambda x: x['implementation'] == 'reduction' and x['bindstrat'] == 1, data), "compare_redu_1_seqtime.png", "Clusters", "Clusters comparison")