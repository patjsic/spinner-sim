import numpy as np
import argparse
from scipy import misc
from scipy.signal import argrelmax
import scipy.ndimage as imglib
from math import factorial
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description="")
parser.add_argument("-nframes", type=int, help="number of time steps to analyse")
parser.add_argument("-fps", type=int, help="frame rate per second")
parser.add_argument("-f", type=str, help="root name of image files to analyse")
args = parser.parse_args()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', weight='bold')
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)

nframes = args.nframes
filename = args.f
fps = args.fps
# print nframes, fps

data_grey = []
for i in range(1, nframes+1):
    data_grey.append( misc.imread(filename + '-{:03d}.jpg'.format(i), flatten=True) )
data_grey = np.array(data_grey)
X = data_grey.shape[1]
Y = data_grey.shape[2]
# print data_grey.shape, X, Y
# plt.imshow(data_grey[55], cmap="gray", origin="lower")
# plt.show()
# exit(1)

data_smooth = []
for i in range(nframes):
    data_smooth.append( imglib.gaussian_filter(data_grey[i], sigma = 5) )
data_smooth = np.array(data_smooth)
# plt.imshow(data_smooth[55], cmap="gray", origin="lower")
# plt.show()

# remove global average
data_smooth -= np.mean(data_smooth)

# remove site specific mean
data_t = data_smooth - np.mean(data_smooth, axis=0)

step = 1
data_t = data_t[0:nframes:step, :, :]
# print data_t.shape
# exit(1)
tcorr = np.zeros(nframes/step)
for i in range(X):
    for j in range(Y):
        pixel = data_t[:, i, j]
#         print pixel.shape
        ACF = np.correlate(pixel, pixel, mode='full')
        tcorr += ACF[ACF.size/2:]

time_arr = np.linspace(0, nframes, nframes/step) / fps
# print out time-correlation data
tcorr /= tcorr[0]
# print '#variance from correlation calc {}'.format(tcorr[0])

# write time correlation of mean angle to file
outfile = "time-corr-" + filename
f = open(outfile.replace('wmv', 'txt'), "w")
for i in range(len(time_arr)):
    f.write('{} {}\n'.format(time_arr[i], tcorr[i]))
f.close()

# calculate the maxima of the correlation function to get
# the rotational period

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

tcorr_cull = tcorr[np.where(time_arr >= 1.0)]
tcorr_smooth_cull = savitzky_golay(tcorr_cull, window_size=15, order=7)
tcorr_smooth = np.concatenate( (tcorr[np.where(time_arr < 1.0)], tcorr_smooth_cull) )
max_indices = argrelmax(tcorr_smooth)[0]
period = 2.0 * np.mean( np.diff(time_arr[max_indices]) )
# print time_arr[max_indices]
f = open('period-val.txt', "w")
f.write('period of rotation is {} seconds.'.format(period))
f.close()

plt.plot(time_arr, tcorr, 'bv', label='original correlation')
plt.plot(time_arr, tcorr_smooth, 'ro', label='smoothed correlation')
plt.plot(time_arr, tcorr_smooth, 'r', linewidth=4, alpha=0.5)
for i in np.arange(0, nframes/fps, period):
    plt.vlines(i, min(tcorr_smooth), 1.0, 'k', linestyle='dashed', linewidth=2)
# plt.show()
plt.xlabel('time (in seconds)', fontsize=28)
plt.ylabel('correlation', fontsize=28)
plt.ylim(min(tcorr)-0.1,0.7)
plt.legend(loc='upper right', fontsize=28)
plt.tight_layout()
plt.savefig(outfile.replace('wmv', 'png'))

# plt.subplot(121)
# plt.semilogy(time_arr, tcorr, 'ro')
# plt.semilogy(time_arr, tcorr, 'r', linewidth=4, alpha=0.5)
# plt.subplot(122)
# plt.plot(time_arr, tcorr, 'ro')
# plt.plot(time_arr, tcorr, 'r', linewidth=4, alpha=0.5)
# plt.tight_layout()
# plt.show()

# ft_tcorr = np.abs( np.fft.fftshift(np.fft.fft(tcorr)) )
# k_arr = np.fft.fftshift( np.fft.fftfreq( len(tcorr), d=nframes/fps ) )
# plt.plot(k_arr, ft_tcorr, 'ro')
# plt.plot(k_arr, ft_tcorr, 'r', linewidth=2, alpha=0.4)
# plt.show()
