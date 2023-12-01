import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

width = 0.25
br = np.arange(7)

def plot_threads_time(measurement):
  threads = []
  times = []
  for ms in measurement['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      times.append(0)
    else:
      times.append(ms['time'])

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.bar(br, times, width, label='Threads-Time')
  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Time', fontweight='bold', fontsize=14)
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
      seqtimes.append(0)
    else:
      seqtimes.append(seqtime / ms['time'])

  fig, ax = plt.subplots(figsize=(10, 6))
  plt.bar(br, seqtimes, width, label='Threads-Speedup')
  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(measurement['implementation'].capitalize())
  plt.title("Bindstrat :" + str(measurement['bindstrat']) + ", Clusters " + str(measurement['clusters']), y=1.02)
  plt.grid(True)
  ax.set_xticks(np.arange(0, 7, 1))
  ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])

  filename = measurement['implementation'] + '_' + str(measurement['bindstrat']) + '_' + str(measurement['clusters']) + '_' + 'seqtime' + '.png'
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)

  plt.close()

width = 0.1

def plot_simple(measurement, offset):
  threads = []
  times = []
  for ms in measurement['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      times.append(0)
    else:
      times.append(ms['time'])

  # plt.plot(times, marker='o', label=f":{measurement['bindstrat']}")
  plt.bar(offset, times, width, label=f"{measurement['bindstrat']}")


def plot_critical(measurements):
  fig, ax = plt.subplots(figsize=(10, 6))
  br = np.arange(7)

  for measurement in measurements:
    plot_simple(measurement, br)
    br = [x + width for x in br]

  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Time', fontweight='bold', fontsize=14)
  plt.suptitle(measurement['implementation'].capitalize())
  plt.title("Clusters " + str(measurement['clusters']))
  plt.legend(title='Bindstrat')
  plt.grid(True)
  # ax.set_xticks(np.arange(0, 7, 1))
  # ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  plt.xticks([r + 0.3 for r in range(7)], ['1', '2', '4', '8', '16', '32', '64'])


  filename = measurement['implementation'] + '_' + 'full' + '_' + str(measurement['clusters']) + '_' + 'time' + '.png'
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()


with open(sys.argv[1], 'r') as file:
  data = json.load(file)
  for measurement in data:
    plot_threads_time(measurement)
    plot_threads_seqtime(measurement)

  plot_critical(filter(lambda ms: ms['implementation'] == 'critical' and ms['size'] == 256, data))

print("Done")
