# rough flagging stuff
filtwin = 30
rough_thresh = 5
max_occupancy = 0.25

medfilt = median_filter(logdata, [1,filtwin])
corrected = logdata - medfilt

MAD = np.median(np.abs(corrected))

rough_flags = (np.abs(corrected) > rough_thresh*MAD).astype(int)

maxfilted = maximum_filter(rough_flags,[1, filtwin]) # get rid of small gaps to the left
opened_flags = minimum_filter(maxfilted,[1, filtwin]) # bring highest flagged channel back down to where the signal really is
# now essentially filled out gaps in cruft then wiped the overhang away on the right edge (an "opening" filter)

lowest_freqs = np.argmin(opened_flags, axis=1) # find where the cruft ends

occupancy = np.sum(opened_flags, axis=0) / opened_flags.shape[0]

lowest_freq = np.min(np.where(occupancy < max_occupancy))

