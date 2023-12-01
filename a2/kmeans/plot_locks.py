import sys
import os
import json
import matplotlib.pyplot as plt
import numpy as np

width = 0.1

def plot_time(tests, filename, legend, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  br = np.arange(7)

  for test in tests:
    plot_util(test, legend, br)
    br = [x + width for x in br]

  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Time', fontweight='bold', fontsize=14)
  plt.suptitle(title)
  plt.title(" Bindstrat :" + str(test['bindstrat']))
  plt.legend(title = legend)
  plt.grid(True)
  # ax.set_xticks(np.arange(0, 7, 1))
  # ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
  plt.xticks([r + 0.3 for r in range(7)], ['1', '2', '4', '8', '16', '32', '64'])

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

def plot_util(test, label, offset):
  threads = []
  times = []
  for ms in test['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      times.append(0)
    else:
      times.append(ms['time'])

  # plt.plot(times, marker='o', label=f"{test[label.lower()]}")
  plt.bar(offset, times, width, label=f"{test[label.lower()]}")

def plot_util_seq(test, label, offset):
  threads = []
  times = []
  seqtime = test['seqtime']
  for ms in test['data']:
    threads.append(ms['threads'])
    if ms['time'] == None:
      times.append(0)
    else:
      times.append(seqtime / ms['time'])

  # plt.plot(times, marker='o', label=f"{test[label.lower()]}")
  plt.bar(offset, times, width, label=f"{test[label.lower()]}")

def plot_seqtime(tests, filename, legend, title):
  fig, ax = plt.subplots(figsize=(12, 8))
  br = np.arange(7)

  for test in tests:
    plot_util_seq(test, legend, br)
    br = [x + width for x in br]

  plt.xlabel('Threads', fontweight='bold', fontsize=14)
  plt.ylabel('Speedup', fontweight='bold', fontsize=14)
  plt.suptitle(title)
  plt.title(" Bindstrat :" + str(test['bindstrat']))
  plt.legend(title = legend)
  plt.grid(True)

  savepath = os.path.join('plots', filename)
  plt.savefig(savepath)
  plt.close()

with open(sys.argv[1]) as f:
  data = json.load(f)

plot_time(filter(lambda x: x['size'] == 32 and x['bindstrat'] == 1, data), "compare_locks_32_1_time.png", "Implementation", "Locks")
plot_time(filter(lambda x: x['size'] == 32 and x['bindstrat'] == 2, data), "compare_locks_32_2_time.png", "Implementation", "Locks")

plot_seqtime(filter(lambda x: x['size'] == 32 and x['bindstrat'] == 1, data), "compare_locks_32_1_seqtime.png", "Implementation", "Locks")
plot_seqtime(filter(lambda x: x['size'] == 32 and x['bindstrat'] == 2, data), "compare_locks_32_2_seqtime.png", "Implementation", "Locks")
