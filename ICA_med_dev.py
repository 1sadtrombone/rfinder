import SNAPfiletools as sft
import numpy as np
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/tajdyson/MARS/data_auto_cross"
sensitivity = 5 # anything this*MAD above/below median flagged

# HARDCODED data times: night of 9th
ti = 1562814420
tf = ti + 3600 * 8

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!
good = subject

u, s, v = np.linalg.svd(np.log10(good), 0)
s[2:] = 0
first_two = np.matmul(u, np.matmul(np.diag(s), v))
corrected = np.log10(good) - first_two

# make it linear again before median?

median = np.median(corrected, axis=0)
minus_med = corrected - median
MAD = np.median(np.abs(minus_med), axis=0)

rfi_removed = good.copy()
rfi_rem_min_med = minus_med.copy()
flags = np.where(np.abs(minus_med) > sensitivity * MAD, np.ones(good.shape), np.zeros(good.shape))
rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]
rfi_removed[np.where(flags)] = np.nan
rfi_rem_min_med[np.where(flags)] = np.nan

plt.figure()
plt.imshow(first_two, aspect='auto', vmin=7, vmax=10)

#freqs = [63, 227, 547, 1722]
freqs = []
for freq in freqs:
    plt.figure()
    plt.title(freq)
    plt.plot(minus_med[:,freq])
    plt.plot(minus_med[:,freq])
    plt.plot([MAD[freq]*sensitivity]*good.shape[0])
    plt.plot([-MAD[freq]*sensitivity]*good.shape[0])

#plt.figure()
#plt.title('minus med')
#plt.imshow(minus_med, aspect='auto', vmin=-1e-2, vmax=1e-2)
#plt.figure()
#plt.title('minus svd')
#plt.imshow(corrected, aspect='auto', vmin=-1e-2, vmax=1e-2)

#plt.figure()
#plt.title('rfi removed, log scale')
#plt.imshow(np.log10(rfi_removed), aspect='auto', vmin=7, vmax=10)

#plt.figure()
#plt.title('rfi removed, minus med')
#plt.imshow(rfi_rem_min_med, aspect='auto', vmin=-1e-2, vmax=1e-2)

plt.figure()
plt.title('log scale')
plt.imshow(np.log10(good), aspect='auto', vmin=7, vmax=10)

"""

plt.figure()
index = 300
plt.title(index)
plt.plot(minus_med[:,index])
plt.plot(minus_med.shape[0]*[sensitivity*MAD[index]])
plt.plot(minus_med.shape[0]*[sensitivity*-MAD[index]])


f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3]})
a1.set_axis_off()

a0.plot(rfi_occ_freq)
a0.margins(0)
a3.plot(rfi_occ_time, np.linspace(0, len(rfi_occ_time), len(rfi_occ_time)))
a3.margins(0)
a3.set_ylim(a3.get_ylim()[::-1])
a2.imshow(np.log10(rfi_removed), aspect='auto', vmin=10, vmax=16)
plt.tight_layout()
"""
plt.show()

