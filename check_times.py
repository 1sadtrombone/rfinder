import numpy as np
import matplotlib.pyplot as plt
import SNAPfiletools as sft

times_file = "/media/sf_MARS/logs/good_times.csv"
data_dir = "/media/sf_MARS/data_auto_cross"

times = np.genfromtxt(times_file, delimiter=',')


for i in [12,13]:

    ti = times[2*i]
    tf = times[2*i+1]

    time, data = sft.ctime2data(data_dir, ti, tf)

    plt.imshow(np.log10(data[0]), aspect='auto')
    plt.show()
    
