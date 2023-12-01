import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

width = 0.2

def plot_simple(measurement, offset):
  threads = []
  times = []
  for ms in measurement['data']:
    threads.append(ms['threads'])
    times.append(ms['time'])

  plt.bar(offset, times, width, label=f"{measurement['block']}")
  # plt.plot(times, marker='o', label=f":{measurement['block']}")

def plot_fw(measurements):
  fig, ax = plt.subplots(figsize=(10, 6))
  br = np.arange(7)

  for measurement in measurements:
    plot_simple(measurement, br)
    br = [x + width for x in br]

  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Time', fontweight='bold', fontsize=14)
  plt.suptitle('Floyd-Warshall')
  plt.title(str(measurement['size']) + 'x' + str(measurement['size']))
  plt.legend(title='Block Size')
  plt.grid(True)
  # ax.set_xticks(np.arange(0, 7, 1))
  # ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  plt.xticks([r + 0.2 for r in range(7)], ['1', '2', '4', '8', '16', '32', '64'])


  filename = f"fw_{measurement['size']}.png"
  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()


with open(sys.argv[1], 'r') as file:
  data = json.load(file)

plot_fw(filter(lambda ms: ms['size'] == 1024, data))
plot_fw(filter(lambda ms: ms['size'] == 2048, data))
plot_fw(filter(lambda ms: ms['size'] == 4096, data))

print('Done')
