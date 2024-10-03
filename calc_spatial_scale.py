import matplotlib.pyplot as plt
import Utils as ut
import numpy as np
import cv2
from pathlib import Path
import tracemalloc
# change font size
plt.rcParams.update({'font.size': 20})

def preProcess(f_name=None, med_ksize=5):
    data = Path("./Data/" + f_name).resolve()
    n = data
    thresh = 100

    # Load an color image.
    shot = cv2.imread(filename=str(data))
    # sigma_est = estimate_sigma(shot, multichannel=True, average_sigmas=True)

    # Apply histogram equalization [CLAHE]
    # shot = enhance(shot, clip_limit=3)
    # image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

    # Adjust Contrast
    image_yuv = cv2.cvtColor(shot, cv2.COLOR_BGR2YUV)

    # Convert to RGB
    shot_nl = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    # Split Channels
    (b, g, r) = (shot_nl[:, :, 0], shot_nl[:, :, 1], shot_nl[:, :, 2])
    g = cv2.medianBlur(g, med_ksize)
    g = cv2.GaussianBlur(g, (9, 9), 0)
    shot = []
    return b, g, r

def downsample_2d(image, freq=1):
    [m, n] = image.shape

    # Down sampling
    f = freq

    # Create a matrix of all zeros for
    dsim = np.zeros((m // f, n // f), dtype=np.int)

    # Assign the down sampled values from the original
    # image according to the down sampling frequency.
    for i in range(0, m, f):
        for j in range(0, n, f):
            try:
                dsim[i // f][j // f] = image[i][j]
            except IndexError:
                pass
    return dsim

def compute_fft_image(mat):
    # displaying the memory
    print("compute_fft_image: "+str(tracemalloc.get_traced_memory()))
    # F = np.fft.fft2(mat)
    # fshift = np.fft.fftshift(F.copy())
    fshift = np.fft.fftshift(np.fft.fft2(mat))
    # fshift = sp.fftpack.fft2(mat, overwrite_x=True)
    F=0
    return fshift


def pad_and_fft(lst):
    '''
    Pad each matrix in lst with nearest 2^N zeros and then compute FFT.  They should be 2D matrices as we are computing
    a 2D FFT.
    :param lst: tuple of matrices to compute FFT on.
    :return:
    '''
    orig_size = lst[1].shape
    ret_lst = []
    for l in lst:
        # displaying the memory
        print("pad_and_fft: " + str(tracemalloc.get_traced_memory()))
        ret_lst.append(compute_fft_image(mat=pad_image_po2(l)))
    return ret_lst, orig_size

def pad_image_po2(image):
    '''
    Pad image with zeros to conform to 2^N requirement for FFTs.
    :param image: Image to pad with zeros.
    :return: Padded image
    '''
    pots = [2 ** exp for exp in range(16)] #powersoftwo
    height, width = get_dimensions(image.shape, pots)
    if width == None or height == None:
        print("'%s' has too large dimensions, skipping.")
    w_diff = width-image.shape[1]
    h_diff = height-image.shape[0]

    if w_diff%2==0 and h_diff%2==0:
        zp_im = np.pad(image, ((int(h_diff/2), int(h_diff/2)), (int(w_diff/2), int(w_diff/2))), 'constant') # (top, bottom),(left, right)
    elif w_diff%2==0 and h_diff != 0:
        zp_im = np.pad(image, ((int(h_diff/2)+1, int(h_diff/2)), (int(w_diff/2), int(w_diff/2))), 'constant') # (top, bottom),(left, right)
    elif w_diff%2!=0 and h_diff == 0:
        zp_im = np.pad(image, ((int(h_diff/2), int(h_diff/2)), (int(w_diff/2)+1, int(w_diff/2))), 'constant') # (top, bottom),(left, right)
    elif w_diff%2!=0 and h_diff != 0:
        zp_im = np.pad(image, ((int(h_diff/2)+1, int(h_diff/2)), (int(w_diff/2)+1, int(w_diff/2))), 'constant') # (top, bottom),(left, right)

    return zp_im

def get_dimensions(size, pots):
    '''Returns closest greater or equal than power-of-two dimensions.

    If a dimension is bigger than max(pots), that dimension will be returned
    as None.

    '''
    width, height = None, None
    for pot in pots:
        # '<=' means dimension will not change if already a power-of-two
        if size[0] <= pot:
            width = pot
            break
    for pot in pots:
        if size[1] <= pot:
            height = pot
            break
    return width, height

def plot_one_thing(M, f_name='name', plot_3d=False, colorbar=True, vmax=None):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        im = ax0.imshow(M, cmap="RdBu", vmin=vmax[0], vmax=vmax[1])
    else:
        im = ax0.imshow(M, cmap="RdBu")
    plt.tight_layout()
    ax0.set_title(f_name)
    if colorbar:
        plt.colorbar(im)
    plt.show()
    plt.close(fig)

    if plot_3d:
        # create the x and y coordinate arrays (here we just use pixel indices)
        xx, yy = np.mgrid[0:M.shape[0], 0:M.shape[1]]

        # create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, M, rstride=1, cstride=1, cmap=plt.cm.plasma, linewidth=0)

        # show it
        plt.show()
        plt.close(fig)

# Function to convert RGB image to grayscale
def rgb2gray(image):
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])


# Smoothing function using a moving average
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')

# Smoothing function using Gaussian filter
def gaussian_smoothing(signal, sigma=1):
    kernel_size = int(6 * sigma + 1)
    kernel = np.exp(-np.linspace(-(kernel_size//2), kernel_size//2, kernel_size)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel
    return np.convolve(signal, kernel, mode='same')

# Function to compute gradients using Sobel filters
def sobel_filter(signal):
    sobel_x = np.array([1, 0, -1])  # Simplified 1D Sobel filter for gradients
    gradient_x = np.convolve(signal, sobel_x, mode='same')
    return gradient_x

# Non-maximum suppression: Thins out edges by only keeping local maxima
def non_maximum_suppression(gradient):
    # Compare each gradient value with its neighbors
    suppressed = np.zeros_like(gradient)
    for i in range(1, len(gradient) - 1):
        if gradient[i] > gradient[i - 1] and gradient[i] > gradient[i + 1]:
            suppressed[i] = gradient[i]
    return suppressed

# Double thresholding and edge tracking by hysteresis
def hysteresis_thresholding(gradient, low_threshold, high_threshold):
    edges = np.zeros_like(gradient)
    strong = gradient > high_threshold
    weak = (gradient > low_threshold) & (gradient <= high_threshold)
    
    # Label strong edges as 1, and try to connect weak edges
    edges[strong] = 1
    for i in range(1, len(edges) - 1):
        if weak[i] and (edges[i - 1] == 1 or edges[i + 1] == 1):
            edges[i] = 1
    return edges

# Rounding the uncertainty to 1 or 2 significant figures
def round_uncertainty(value, uncertainty):
    # Round the uncertainty to 1 or 2 significant figures
    if uncertainty < 1:
        uncertainty_rounded = round(uncertainty, -int(np.floor(np.log10(uncertainty))))
    else:
        uncertainty_rounded = round(uncertainty, -int(np.floor(np.log10(uncertainty))) + 1)
    
    # Round the value to match the decimal place of the uncertainty
    value_rounded = round(value, -int(np.floor(np.log10(uncertainty_rounded))))
    
    return value_rounded, uncertainty_rounded

#############################################
######### MAIN Processing ###################
#############################################

f_name_plot = './Plots/'
f_name_dir = 'Quantaray/8_23_2024/'
f_name = f_name_dir+'00008_interferometer'
f_name_h = f_name + ".jpg"
(b, g, r) = preProcess(f_name=f_name_h, med_ksize=1)

# Downsample
f=4
r = downsample_2d(r, freq=f)
g = downsample_2d(g, freq=f)
b = downsample_2d(b, freq=f)

g6 = g[550:640, 1250:1425]

g6_horizontal = np.mean(g6[:, 70:90], axis=1)

# # Apply moving average smoothing

# Apply Gaussian smoothing
sigma = 1
g6_horizontal_smooth = gaussian_smoothing(g6_horizontal, sigma=sigma)

# Step 1: Compute gradient using Sobel filter
gradient = sobel_filter(g6_horizontal_smooth)

# Step 2: Apply non-maximum suppression to thin out edges
nms_gradient = non_maximum_suppression(np.abs(gradient))

# Step 3: Apply hysteresis thresholding
low_threshold = 10
high_threshold = 50
edges = hysteresis_thresholding(nms_gradient, low_threshold, high_threshold)

# Plot the noisy signal, smoothed signal, and detected edges
fig1 = plt.figure()
plt.figure(figsize=(13.385, 13.385))

# Plot original noisy signal
plt.subplot(2, 1, 1)
plt.plot(g6_horizontal, label="Noisy Pixel Values", marker='o', markersize=8, color='blue', alpha=0.7, linewidth=4)
plt.title("Noisy Oscillating Signal")
plt.ylim([80, 230])
# plt.xlabel("Pixel Index")
plt.ylabel("Pixel Value")
plt.xticks([])
# plt.grid(True)

# Plot smoothed signal
plt.subplot(2, 1, 1)
plt.plot(g6_horizontal_smooth, label="Smoothed Signal (Gaussian)", color='green', alpha=0.7, linewidth=4)
plt.ylim([80, 230])
plt.title("Smoothed Signal")
plt.xticks([])
# plt.xlabel("Pixel? Index")
plt.ylabel("Pixel Value")
# plt.grid(True)

# Plot edges detected
plt.subplot(2, 1, 2)
plt.plot(edges, label="Edges Detected (Canny)", marker='o', color='red')
plt.title("Detected Edges (Simplified Canny)")
plt.xlabel("Pixel Index")
plt.ylabel("Edge")
# plt.grid(True)

plt.tight_layout()

# Check if plot directory exists
Path(f_name_plot+f_name_dir).mkdir(parents=True, exist_ok=True)

plt.savefig(f_name_plot+f_name+"_edge_detection.png", dpi=600)
plt.show()

# Find the indices of the detected edges
edge_locations = np.where(edges == 1)[0]
# Detect Rising Edges (where signal goes from 0 -> 1)
rising_edges = edge_locations[1::2]

# Detect Falling Edges (where signal goes from 1 -> 0)
falling_edges = edge_locations[0::2]

# Find difference between every other edge locations
edge_differences_rising = np.diff(rising_edges)
edge_differences_fall = np.diff(falling_edges)

mean_rising = np.mean(edge_differences_rising)
mean_fall = np.mean(edge_differences_fall)

mean = np.mean([mean_rising, mean_fall])  
print("Mean: ", mean)

# G = 0
# E = 6

# lp_p_mm = 2**(G+((E-1)/6))
# mm_p_lp = 1/lp_p_mm
# pixel_p_lp = mean
# mm_p_pixel = mm_p_lp/pixel_p_lp

# print("mm per pixel: ", mm_p_pixel)

# Calculate the standard deviation of the edge differences
std_rising = np.std(edge_differences_rising, ddof=1)
std_falling = np.std(edge_differences_fall, ddof=1)

# Combine the standard deviations
std_edge_diff = np.mean([std_rising, std_falling])

# Number of edge differences
N_rising = len(edge_differences_rising)
N_falling = len(edge_differences_fall)
N_total = N_rising + N_falling

# Calculate the standard error of the mean
standard_error_mean = std_edge_diff / np.sqrt(N_total)

# Parameters for calculating mm/pixel
G = 2  # Grid factor
E = 6  # Element factor

lp_p_mm = 2**(G + (E - 1) / 6)  # Line pairs per mm
mm_p_lp = 1 / lp_p_mm  # mm per line pair
pixel_p_lp = mean
mm_p_pixel = mm_p_lp / pixel_p_lp  # mm per pixel

# Uncertainty propagation for mm/pixel
uncertainty_mm_per_pixel = (mm_p_lp / (mean ** 2)) * standard_error_mean

# Use the function to round the mm per pixel and its uncertainty
mm_p_pixel_rounded, uncertainty_mm_per_pixel_rounded = round_uncertainty(mm_p_pixel, uncertainty_mm_per_pixel)

# Print the result in the format "value ± uncertainty"
print(f"mm per pixel: {mm_p_pixel_rounded} ± {uncertainty_mm_per_pixel_rounded}")

# # Pad with zeros to 2^N #
# ((F_s, F_b), orig_size) = pad_and_fft((g[560:640, 1250:1425], b[560:640, 1250:1425]))

# Plot the original image

fig1 = plt.figure()
plt.figure(figsize=(13.385, 13.385))
plt.imshow(g6, cmap='gray') 
plt.savefig(f_name_plot+f_name+"_original_image.png", dpi=600)
plt.show()

# l0 = 0.79375 #inch
# dx = l0/(1390-713)