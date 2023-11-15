import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_threads_time(measurement):
  threads = []
  times = []
  for ms in measurement['data']:
    threads.append(ms['threads'])
    times.append(ms['time'])

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.plot(times, marker='o', label='Threads-Time')
  plt.xlabel('Threads')
  plt.ylabel('Time')
  plt.suptitle(measurement['implementation'].capitalize())
  plt.title("Bindstrat :" + str(measurement['bindstrat']) + ", Clusters " + str(measurement['clusters']))
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])

  filename = measurement['implementation'] + '_' + str(measurement['bindstrat']) + '_' + str(measurement['clusters']) + '_' + 'time' + '.png'
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  
  plt.close()

def plot_threads_seqtime(measurement):
  threads = []
  seqtimes = []
  seqtime = measurement['seqtime']
  for ms in measurement['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      seqtimes.append(None)
    else:
      seqtimes.append(seqtime / ms['time'])

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.plot(seqtimes, marker='o', label='Threads-Speedup')
  plt.xlabel('Threads')
  plt.ylabel('Speedup')
  plt.suptitle(measurement['implementation'].capitalize())
  plt.title("Bindstrat :" + str(measurement['bindstrat']) + ", Clusters " + str(measurement['clusters']), y=1.02)
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])

  filename = measurement['implementation'] + '_' + str(measurement['bindstrat']) + '_' + str(measurement['clusters']) + '_' + 'seqtime' + '.png'
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  
  plt.close()

def plot_simple(measurement):
  threads = []
  times = []
  for ms in measurement['data']:
    threads.append(ms['threads'])
    times.append(ms['time'])

  plt.plot(times, marker='o', label=f":{measurement['bindstrat']}")
  

def plot_critical(measurements):
  fig, ax = plt.subplots(figsize=(10, 6))

  for measurement in measurements:
    plot_simple(measurement)

  plt.xlabel('Threads')
  plt.ylabel('Time')
  plt.suptitle(measurement['implementation'].capitalize())
  plt.title("Clusters " + str(measurement['clusters']))
  plt.legend(title='Bindstrat')
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  
  
  filename = measurement['implementation'] + '_' + 'full' + '_' + str(measurement['clusters']) + '_' + 'time' + '.png'
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()


with open(sys.argv[1], 'r') as file:
  data = json.load(file)
  for measurement in data:
    plot_threads_time(measurement)
    plot_threads_seqtime(measurement)

  plot_critical(filter(lambda ms: ms['implementation'] == 'critical', data))

print("Done")