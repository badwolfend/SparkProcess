import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from scipy.fftpack import fft, dct, idct
from numpy.linalg import norm
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy import stats
# from scipy.misc import imsave
from PIL import Image
import tracemalloc
from pathlib import Path
import cv2
from scipy.ndimage import convolve
from scipy.integrate import trapz, quad
from scipy.ndimage.interpolation import rotate

# starting the monitoring
tracemalloc.start()

def Gs(x, y, s):
    return 1/s/np.sqrt(2*np.pi) * np.exp(-(x**2+y**2) / 2 / s**2)

def Gsx(x, y, s):
    # dx = np.ones_like(x)
    dx=1
    return (Gs(x+dx, y, s)-Gs(x, y, s))/dx

def Gs1d(x, s):
    return 1/s/np.sqrt(2*np.pi) * np.exp(-(x**2) / 2 / s**2)

def Gsx1d(x, s):
    dx=1
    return (Gs1d(x+dx, s)-Gs1d(x, s))/dx

def sample_kernel(fnc, s):
    N = np.ceil(5*s).astype(int)
    xx, yy = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
    # x = np.arange(-N, N+1, dx)
    # y = np.arange(-N, N+1, dx)
    return fnc(xx, yy, s)

def sample_kernel1d(fnc, s):
    N = np.ceil(5*s).astype(int)
    x = np.arange(-N, N+1, 1)

    return fnc(x, s)

def find_critical_dens(wl):
    # wl in micron #
    # dens in /m^3
    n_crit = (10**6)*(1.1e21)/(wl**2)
    print(n_crit)
    Rh_ratio = n_crit
    return Rh_ratio

def save_mat_to_npz(mat, name):
    np.save(name, mat)
    # imsave(loc, mat)


def save_mat_to_im(mat, name):
    form = (mat * 255 / np.max(mat)).astype('uint8')
    im = Image.fromarray(form)
    im.save(name)
    # imsave(name, mat)


def dct2(block):
    return dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')

def idct2(block):
    return idct(idct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')

def poisson(Nx, Ny, rhs):

    # create the domain -- cell-centered finite-difference / finite-volume
    xmin = 0.0
    xmax = 1.0

    ymin = 0.0
    ymax = 1.0

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny

    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy

    x2d = np.repeat(x, Ny)
    x2d.shape = (Nx, Ny)

    y2d = np.repeat(y, Nx)
    y2d.shape = (Ny, Nx)
    y2d = np.transpose(y2d)

    # create the RHS
    # f = rhs/(Nx*Nx)
    f = rhs

    # DCT of RHS ensuring Neumann BC
    Fxy = dct2(f)

    # Compute UW Phase Fourier Transform
    # First define Mesh-Grid for mask #
    gxx, gyy = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny))
    fac = 1/(2*(np.cos(math.pi*gxx/Nx)+np.cos(math.pi*gyy/Ny)-2))

    Fp = np.multiply(Fxy, fac)
    Fp[0, 0]=0

    # transform back to real space
    fsolution = idct2(Fp)

    return fsolution

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def compute_fft_image(mat):
    # displaying the memory
    print("compute_fft_image: "+str(tracemalloc.get_traced_memory()))
    fshift = np.fft.fftshift(np.fft.fft2(mat))
    # fshift  =np.fft.fft2(mat)
    # plot_one_thing(mat, "padded")
    plot_one_thing(np.log10(np.abs(fshift)), "fft", vmax=(0, 8))
    return fshift


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


def compute_ifft_image(mat, shift=False):
    F = mat
    if shift:
        F = np.fft.fftshift(mat)
    f = np.fft.ifft2(F, norm="ortho")
    matrix = np.arctan2(np.imag(f), np.real(f)) + math.pi
    F = 0
    return f, matrix


def GaussianFilter(F=fft, sigx=10, sigy=10):
    cy, cx = F.shape[0] / 2, F.shape[1] / 2
    x = np.linspace(0, F.shape[1], F.shape[1])
    y = np.linspace(0, F.shape[0], F.shape[0])
    X, Y = np.meshgrid(x, y)
    gmask = np.exp(-(((X - cx) / sigx) ** 2 + ((Y - cy) / sigy) ** 2))
    return gmask * F


def wrap2Pi(m):
    p=copy.deepcopy(m)
    p[p > np.pi] = np.mod(p[p > np.pi] + np.pi, 2 * np.pi) - np.pi
    p[p < -np.pi] = np.mod(p[p < -np.pi] + np.pi, 2 * np.pi) - np.pi
    return p

def transform(mat, W):
    # Compute Difference of matrix, but consider that we may need to not use center valleys
    gx = np.diff(mat, axis=1)
    gy = np.diff(mat, axis=0)

    # Padd with Zeros #
    # Columns for dx #
    zxc = np.array(np.zeros(mat.shape[0])).reshape(mat.shape[0], 1)
    zyc = np.array(np.zeros(mat.shape[1])).reshape(1, mat.shape[1])

    gy = np.vstack([gy, zyc])
    gx = np.hstack((gx, zxc))

    W2x = np.multiply(W, gx)
    W2y = np.multiply(W, gy)

    # Padd with Zeros #
    # Columns for dx #
    zxc = np.array(np.zeros(mat.shape[0])).reshape(mat.shape[0], 1)
    zyc = np.array(np.zeros(mat.shape[1])).reshape(1, mat.shape[1])
    Wy2 = np.vstack([zyc, W2y])
    Wx2 = np.hstack((zxc, W2x))

    # Second Difference #
    dy = np.diff(Wy2, axis=0)
    dx = np.diff(Wx2, axis=1)

    return dy+dx


def iterateCG(rho_l,W, norm0, eps):
    l=0
    M = rho_l.shape[0]
    N = rho_l.shape[1]
    rho_l_p = np.zeros_like(rho_l)
    z_l_p = np.zeros_like(rho_l)
    phi = np.zeros_like(rho_l)
    error = np.zeros(0)
    while np.count_nonzero(rho_l) > 0:
        z_l = poisson(N, M, rho_l)
        l+=1
        if l==1:
            pk = z_l
        else:
            betal_t = np.sum(np.sum(np.multiply(rho_l, z_l)))
            betal_b = np.sum(np.sum(np.multiply(rho_l_p, z_l_p)))
            beta_l = betal_t/betal_b
            pk = z_l+beta_l*pk

        # Se the current estimates to be the previous ones #
        rho_l_p = rho_l
        z_l_p = z_l

        # Perform our matrix operations #
        qpl = transform(pk, W)
        alt = np.sum(np.multiply(rho_l, z_l))
        alb = np.sum(np.multiply(pk, qpl))
        al = alt/alb
        phi = phi + al*pk
        rho_l = rho_l-al*qpl
        err = np.linalg.norm(rho_l)
        error = np.append(error, err)

        # Stopping condition check for residual below specified value or if NaN appears#
        if (l >= rho_l.size) or (err < eps * norm0):
            break
        if math.isnan(err):
            print('Unsuccessful.  Cannot interpret NaN value ... ')
            break
    return phi, l, error


def modifiedCGSolver(mat, eps, W, to_plot=False):
    # Add the smoothing from derivatives #
    pij, u_star= compute_rho_ij(mat)
    W = np.divide(W, u_star)

    # Compute Difference of matrix, but consider that we may need to not use center valleys
    gx = np.diff(mat, axis=1)
    gy = np.diff(mat, axis=0)
    gx = wrap2Pi(gx)
    gy = wrap2Pi(gy)

    # Padd with Zeros #
    # Columns for dx #
    zxc = np.array(np.zeros(mat.shape[0])).reshape(mat.shape[0], 1)
    zyc = np.array(np.zeros(mat.shape[1])).reshape(1, mat.shape[1])

    gy = np.vstack([gy, zyc])
    gx = np.hstack((gx, zxc))

    W2 = np.multiply(W, W)
    Wy = np.multiply(W2, gy)
    Wx = np.multiply(W2, gx)
    Wy2 = np.vstack([zyc, Wy])
    Wx2 = np.hstack((zxc, Wx))

    # Second Difference #
    rho_lxx = np.diff(Wx2, axis=1)
    rho_lyy = np.diff(Wy2, axis=0)
    rho_l = rho_lxx+rho_lyy

    norm = np.linalg.norm(rho_l)

    print("Beginning iterations: ")
    phi, iter, error = iterateCG(rho_l, W2, norm, eps)
    print("iter = ", iter, ", Final Error: ", error[error.size-1])
    if to_plot:
        fig = plt.figure(figsize=(12, 14))
        ax0 = fig.add_subplot(1, 1, 1)
        ax0.plot(np.log10(error), 'k', linewidth=4)
        ax0.set(xlabel='Iteration', ylabel='Residual [dB]',
               title='Masked Conjugate-Gradient Convergence')
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print("End Iterations!")
    return phi, iter, error[error.size-1], np.multiply(W, mat)


def compute_rho_ij(mat):
    # Compute Difference of matrix, but consider that we may need to not use center valleys
    gx = np.diff(mat, axis=1)
    gy = np.diff(mat, axis=0)
    gx = wrap2Pi(gx)
    gy = wrap2Pi(gy)

    # Padd with Zeros #
    # Columns for dx #
    zxc = np.array(np.zeros(mat.shape[0])).reshape(mat.shape[0], 1)
    zyc = np.array(np.zeros(mat.shape[1])).reshape(1, mat.shape[1])

    gy = np.vstack([gy, zyc])
    gy = np.vstack([zyc, gy])
    gx = np.hstack((gx, zxc))
    gx = np.hstack((zxc, gx))

    # Second Difference #
    gxx = np.diff(gx, axis=1)
    gyy = np.diff(gy, axis=0)

    # Compute Average Between Gradient #
    gxx2 = np.square(gxx)
    gyy2 = np.square(gyy)
    gxyav = (gxx2+gyy2)/2
    u = gxyav/(gxyav.max())

    rij = gxx + gyy
    u_star = (np.ones(rij.shape)+u)
    rij_star = np.multiply(u_star, rij)

    return rij_star, u_star

def compute_wrapped_phase(lst):
    '''
    Compute the wrapped phase given the demodulated complex matrix.  This is an atan2 function, so it is modulo 2PI.
    This particular instance adds PI to the output to make it -PI to +PI.  This is for convenience. It doesn't make
    any difference.
    :param lst: list of demodulated phases to compute wrapped phase of.
    :return: list of wrapped phase matrices
    '''
    ret_list = []
    for l in lst:
        ret_list.append(np.arctan2(l.imag, l.real) + math.pi)
    return ret_list

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
        ret_lst.append(compute_fft_image(mat=l))
        # ret_lst.append(compute_fft_image(mat=pad_image_po2(l)))
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


def rm_pad_image_po2(image, size):
    '''
    Remove the 2^N padding on image.
    :param image:  Image to remove padding from.
    :param size: Original Size of image.
    :return:
    '''
    cx, cy = int(image.shape[1]/2), int(image.shape[0]/2)
    top = cy - int(size[0]/2)
    bot = cy + int(size[0]/2)
    left = cx - int(size[1]/2)
    right = cx + int(size[1]/2)
    rzp_im = image[top:bot, left:right]
    return rzp_im


def resize_list_images(lst, size):
    '''
    Undo the 2^N padding on each object in lst.
    :param lst: List of matrices to undo the padding on.
    :param size: Size of the original matrix
    :return:
    '''
    ret_list = []
    for l in lst:
        ret_list.append(rm_pad_image_po2(l, (size[0], size[1])))
    return ret_list


def resize_image(coords, size, mat):
    '''
    Resize image.
    :param coords:
    :param size:
    :param mat:
    :return:
    '''
    # plt.imshow(mat[coords[0]:coords[0] + size[0], coords[1]:coords[1] + size[1]])
    # plt.show()
    return mat[coords[0]:coords[0] + size[0], coords[1]:coords[1] + size[1]]


def gen_mask(rows, cols, x, y, typ='box', kxsize=3, kysize=3):
    '''
    Generate a mask with 1s in the region to keep.  0s elsewhere.  Assumes horizontal fringes and that you will only
    pass one lobe of the carrier.  This should be improved later.
    :param rows: Total rows of mask
    :param cols: Total columns of mask
    :param x: maximum freq to pass in x-direction
    :param y: tuple of (max min) freq to pass in y-direction
    :param typ: string of the type of window mask to apply (gaussian or box)
    :return: mask of coefficient weights for FFT
    '''
    if typ == 'box':
        mask = np.zeros((rows, cols), np.float_)
        mask[y[0]:y[1], x[0]:x[1]] = 1
        mask = np.rot90(mask, 1)
        mask = mask.T
        mask[y[0]:y[1], x[0]:x[1]] = 1
    elif typ =='gaussian':
        mask = np.zeros((rows, cols), np.float_)
        if kxsize == kysize:
            GK = gaussian_kernel(kxsize, to_norm=False)
        else:
            GK = ob_gaussian_kernel(sigmax=kxsize, sigmay=kysize)
        n, m = GK.shape
        mask[y-n//2:y+n//2+1, x-m//2:x+m//2+1] = GK
    return mask


def apply_mask_ifft(lst, mask):
    '''
    Apply the Mask to the Fourier domain of the FFTs in lst.  This should be a matrix of 1s (or some weighting >0) for
    values to keep in the FFT.  It will be centered around carrier lobes.   Then each will be centered and inverted.
    :param lst:  FFTs to apply mask to and to invert.
    :param mask: Mask of weights for Fourier coefficients
    :return: Inverted complex carrier demodulated image
    '''
    ret_lst = []

    for l in lst:
        # F_m = l*mask
        # displaying the memory
        print("apply_mask_ifft: " + str(tracemalloc.get_traced_memory()))
        # ps = sp.fftpack.ifftshift(sp.fftpack.ifft2(l*mask, overwrite_x=False))
        ps = np.fft.ifftshift(l*mask)

        ret_lst.append((np.fft.ifft2(ps), l*mask))
    return ret_lst


def gen_res_map(lst, sig):
    ret_lst = []
    for l in lst:
        ret_lst.append((residualMap(l, sig)))
    return ret_lst

def residualMap(psi, sig):
    sz = psi.shape
    R = np.zeros((sz[0] - 1, sz[1] - 1))
    RMask = np.ones_like(psi)
    eps = 0.3 # How far from the fringe border to be ... ad hoc ... :(
    for j in range(0, sz[1]-1):
        for i in range(0, sz[0] - 1):
            dyl = psi[i + 1, j] - psi[i, j]
            dxb = psi[i + 1, j + 1] - psi[i + 1, j]
            dyr = psi[i, j + 1] - psi[i + 1, j + 1]
            dxt = psi[i, j] - psi[i, j + 1]
            R[i, j] = dyl + dyr + dxb + dxt
            left = j-1
            top = i-1
            if left>sig and top >sig:
                if (R[i, j] != 0):
                    # if (np.sign(R[i, j])-np.sign(R[i-1, j]) == 0) or (np.sign(R[i, j])-np.sign(R[i, j-1]) == 0):
                    if psi[i,j]> eps and psi[i,j]< 2*np.pi - eps:
                        # RMask[i, j]=0
                        RMask[i-sig:i+sig, j-sig:j+sig]=0
    return R, RMask


def computeCoerrCoeffSNR(s, ps):
    sig_s = np.std(s.flatten())
    sig_ps = np.std(ps.flatten())

    cv = np.cov(s.flatten(), ps.flatten())
    c_hat = cv/(sig_ps*sig_s)

    SNR = np.sqrt(c_hat/(1-c_hat))
    print("Coerrelation estimation: %5.2f, SNR = %5.2f\n" %( c_hat, SNR))

    return c_hat, SNR


def computeSNR(s, sig_n):
    sig_s = np.std(s.flatten())
    SNR = np.sqrt((sig_s/sig_n)**2-1)
    return SNR


def checkNyquist(max, N):
    '''
    Check Nyquist for image Fourier transform.
    :param max: Maximum frequency used in processing
    :param N: Size of image in the direction of the FFT
    :return: True if below Nyquist and False if above.
    '''
    return True if max < N/2 else False

def compute_chi2(y_true, y, err, bins):
    # Plot the results
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.05,
                        bottom=0.1, top=0.95, hspace=0.05)

    ax = fig.add_subplot(2, 2, 1, xticks=[])

    # compute the mean and the chi^2/dof
    z = (y - y_true) / err
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (y_true.size - 1)
    print("chi2dof = ", chi2dof)
    # compute the standard deviations of chi^2/dof
    sigma = np.sqrt(2. / (y_true.size - 1))
    nsig = (chi2dof - 1) / sigma

    # plot the points with errorbars
    ax.errorbar(bins, y, err, fmt='.k', ecolor='gray', lw=1)

    # Add labels and text
    ax.text(0.98, 0.02,
            r'$\chi^2_{\rm dof} = %.2f\, (%.2g\,\sigma)$' % (chi2dof, nsig),
            ha='right', va='bottom', transform=ax.transAxes)
    plt.show()


def fit_to_dist(M, hist_n, hist_bins, dist='Gaussian', to_plot=False):
    x = hist_bins[0:hist_bins.size]
    y = hist_n

    if dist == 'gaussian':
        # calculate binmiddles
        bin_middles = 0.5 * (x[1:] + x[:-1])

        n = len(bin_middles)  # the number of data
        mean = sum(bin_middles * y) / n  # note this correction
        sigma = sum(y * (bin_middles - mean) ** 2) / n  # note this correction

        def gaussian(x, amp, cen, wid):
            "1-d gaussian: gaussian(x, amp, cen, wid)"
            return (amp / (np.sqrt(2 * np.pi) * wid)) * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

        parameters, cov_matrix = curve_fit(gaussian, bin_middles, y)

        # plot poisson-deviation with fitted parameter
        if (to_plot):
            x_plot = np.linspace(bin_middles[0], bin_middles[bin_middles.size - 1], 1000)
            plt.plot(x_plot, gaussian(x_plot, *parameters), 'r-', lw=2)
            plt.show()


    elif dist == 'gamma':
        class Gamma(GenericLikelihoodModel):
            nparams = 3
            def loglike(self, params):
                return stats.gamma.logpdf(self.endog, *params).sum()

        x_g = np.arange(0, 250, .1)
        fig, axes = plt.subplots(1, 1, figsize=(13, 4))
        ax = axes
        n, bins, patches = ax.hist(M, bins=40, normed=True);

        # fit the distributions, get the PDF distribution using the parameters
        shape1, loc1, scale1 = stats.gamma.fit(M)
        bin_m = 0.5 * (bins[1:] + bins[:-1])

        # Do not need to use this fit, but it is useful for plotting #
        g2 = stats.gamma.pdf(x=bin_m, a=shape1, loc=loc1, scale=scale1)

        # Package up parameters so the Gamma fit cam compute statistics
        params = (shape1, loc1, scale1 )
        res = Gamma(M).fit(start_params=params)
        res.hessv
        res.df_model = len(params)
        res.df_resid = len(M) - len(params)
        print(res.summary())


def plot_hist_image(mat, num_bins, normed=True):
    x = mat.flatten()
    if normed:
        x=x/x.max()
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor='blue', alpha=0.5)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
    return n, bins

def plot_four_things(M1, M2, M3, M4):
    fig =plt.figure()
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.axis('off')
    im0 = ax0.imshow(M1,cmap="gray")
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.axis('off')
    im1 = ax1.imshow(M2, cmap="gray")
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.axis('off')
    im2 = ax2.imshow(M3, cmap="gray")
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.axis('off')
    im3 = ax3.imshow(M4, cmap="gray")
    plt.show()
    plt.close(fig)


def plot_two_things(M1, M2, f_name='name', plot_3d=False):
    fig =plt.figure()
    ax0 = fig.add_subplot(2, 1, 1)
    im0 = ax0.imshow(M1,cmap="RdBu")
    ax0.set_title(f_name)
    ax1 = fig.add_subplot(2, 1, 2)
    im = ax1.imshow(M2, cmap="RdBu")
    ax1.set_title(f_name)
    plt.colorbar(im)
    plt.show()
    plt.close(fig)

    if plot_3d:
        # create the x and y coordinate arrays (here we just use pixel indices)
        xx, yy = np.mgrid[0:M1.shape[0], 0:M1.shape[1]]

        # create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, M1, rstride=1, cstride=1, cmap=plt.cm.plasma,
                        linewidth=0)
        ax.plot_surface(xx, yy, M2, rstride=1, cstride=1, cmap=plt.cm.plasma,
                        linewidth=0)

        # show it
        plt.show()
        plt.close(fig)


def plot_one_thing(M, f_name='name', plot_3d=False, colorbar=True, vmax=None, to_show=True, cmap="RdBu"):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        # im = ax0.imshow(M, cmap="RdBu", vmin=vmax[0], vmax=vmax[1])
        im = ax0.imshow(M, cmap=cmap, vmin=vmax[0], vmax=vmax[1])
    else:
        im = ax0.imshow(M, cmap=cmap)
    plt.tight_layout()
    ax0.set_title(f_name)
    if colorbar:
        plt.colorbar(im)
    if to_show:
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
        if to_show:
            plt.show()
            plt.close(fig)

def plot_one_line(x=None, y=None, f_name='name', vmax=None, to_show=True):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        if x is not None:
            im = ax0.plot(x, y)
            ax0.set_xlim(vmax[0], vmax[1])
        else:
            im = ax0.plot(y)
            ax0.set_xlim(vmax[0], vmax[1])

    else:
        if x is not None:
            im = ax0.plot(x, y)
        else:
            im = ax0.plot(y)
    plt.tight_layout()
    ax0.set_title(f_name)

    if to_show:
        plt.show()
        plt.close(fig)


def plot_one_line_with_markers(x=None, y=None, f_name='name', vmax=None, to_show=True):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        if x is not None:
            im = ax0.plot(x, y, 'x')
            ax0.set_xlim(vmax[0], vmax[1])
        else:
            im = ax0.plot(y, 'x')
            ax0.set_xlim(vmax[0], vmax[1])

    else:
        if x is not None:
            im = ax0.plot(x, y, 'x')
        else:
            im = ax0.plot(y, 'x')
    plt.tight_layout()
    ax0.set_title(f_name)

    if to_show:
        plt.show()
        plt.close(fig)


def plot_two_lines(x0=None, y0=None, x1=None, y1=None, f_name='name', vmax=None, to_show=True):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        if x0 is not None:
            im = ax0.plot(x0, y0)
            ax0.plot(x1, y1)
            ax0.set_xlim(vmax[0], vmax[1])
        else:
            im = ax0.plot(y0)
            ax0.plot(y1)
            ax0.set_xlim(vmax[0], vmax[1])

    else:
        if x0 is not None:
            im = ax0.plot(x0, y0)
            ax0.plot(x1, y1)
        else:
            im = ax0.plot(y0)
            ax0.plot(y1)
    plt.tight_layout()
    ax0.set_title(f_name)

    if to_show:
        plt.show()
        plt.close(fig)

def plot_one_line_with_a_marker(x0=None, y0=None, x1=None, y1=None, f_name='name', vmax=None, to_show=True):
    fig =plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    if vmax is not None:
        if x0 is not None:
            im = ax0.plot(x0, y0)
            ax0.plot(x1, y1, 'x', markersize=5)
            ax0.set_xlim(vmax[0], vmax[1])
        else:
            im = ax0.plot(y0)
            ax0.plot(y1, 'x', markersize=5)
            ax0.set_xlim(vmax[0], vmax[1])

    else:
        if x0 is not None:
            im = ax0.plot(x0, y0)
            ax0.plot(x1, y1, 'x', markersize=5)
        else:
            im = ax0.plot(y0)
            ax0.plot(y1, 'x', markersize=5)
    plt.tight_layout()
    ax0.set_title(f_name)

    if to_show:
        plt.show()
        plt.close(fig)

def plot_the_things(g_ps, phase_ps, pps, F_m_ps, F_ps, g_s, phase_s, ps, F_s, F_m_s, pdiff, f_name='name'):
    fig =plt.figure()

    ax0 = fig.add_subplot(2, 4, 1)
    ax1 = fig.add_subplot(2, 4, 2, sharex=ax0, sharey=ax0)
    ax2 = fig.add_subplot(2, 4, 3, sharex=ax0, sharey=ax0)
    ax3 = fig.add_subplot(2, 4, 4)
    ax4 = fig.add_subplot(2, 4, 5, sharex=ax0, sharey=ax0)
    ax5 = fig.add_subplot(2, 4, 6, sharex=ax0, sharey=ax0)
    ax6 = fig.add_subplot(2, 4, 7, sharex=ax0, sharey=ax0)
    ax7 = fig.add_subplot(2, 4, 8, sharex=ax3, sharey=ax3)


    ax0.imshow(g_ps, cmap='gray')
    ax1.imshow(phase_ps, cmap='gray')
    ax2.imshow(pps, cmap='gray', interpolation="nearest")
    ax3.imshow(np.log10(abs(F_m_ps)), cmap='gray', interpolation="nearest", alpha=1)
    ax3.imshow(np.log10(abs(F_ps)), cmap='gray', interpolation="nearest", alpha=0.6)
    ax4.imshow(g_s, cmap='gray')
    ax5.imshow(phase_s, cmap='gray')
    ax6.imshow(ps, cmap='gray', interpolation="nearest")
    ax7.imshow(np.log10(abs(F_s)), cmap='gray', interpolation="nearest", alpha=0.6)
    ax7.imshow(np.log10(abs(F_m_s)), cmap='gray', interpolation="nearest", alpha=1)

    ax0.set_title("Pre-Shot")
    ax1.set_title("Wrapped Phase [Pre-Shot]")
    ax2.set_title("Unwrapped Phase [Pre-Shot]")
    ax3.set_title("Fourier Spectrum [Pre-Shot]")
    ax4.set_title("Shot")
    ax5.set_title("Wrapped-Phase [Shot]")
    ax6.set_title("Unwrapped Phase [Shot]")
    ax7.set_title("Fourier Spectrum [Shot]")

    fig.suptitle(f_name)
    fig =plt.figure()
    im = plt.imshow(pdiff, cmap="gray", interpolation="nearest")
    plt.title(f_name+" Relative Phase")
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def save(self, filepath):
    """ Save model parameters to file.
    """
    all_params = OrderedDict([(param.name, param.get_value()) for param in self.params])
    np.savez(filepath, **all_params)
    print('Saved model to: {}'.format(filepath))


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
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

    # Convert to RGB
    shot_nl = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    # Split Channels
    (b, g, r) = (shot_nl[:, :, 0], shot_nl[:, :, 1], shot_nl[:, :, 2])
    # g = cv2.medianBlur(g, med_ksize)
    # g = cv2.GaussianBlur(g, (9, 9), 0)
    # g = cv2.equalizeHist(g)
    # g=g-np.min(g)
    # g=g/np.max(g)
    # g = g**2
    # g=g*255
    shot = []
    shot_nl=[]
    image_yuv=[]
    return b, g, r

def gaussian_kernel(sigma=3, to_norm=True):
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    f = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2
    x1 = 2 * np.pi * (sigma ** 2)
    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            f[x + m, y + n] = (1 / x1) * x2
    if to_norm:
        f=f/np.sum(f)
    else:
        f=f/np.max(f)
    return f

def ob_gaussian_kernel(sigmax=3, sigmay=3):
    filter_sizex = 2 * int(4 * sigmax + 0.5) + 1
    filter_sizey = 2 * int(4 * sigmay + 0.5) + 1
    f = np.zeros((filter_sizey, filter_sizex), np.float32)
    m = filter_sizex // 2
    n = filter_sizey // 2
    # x1 = 2 * np.pi * (sigma ** 2)
    x1 =1
    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x2 = np.exp(-((x ** 2)/(2 * sigmax ** 2) + (y ** 2)/(2 * sigmay ** 2)) )
            f[y + n, x + m] = (1 / x1) * x2
    f=f/np.max(f)
    f[np.where(f<0)]=0
    return f

def gaussian_filter(M, sigma=3):
    W = gaussian_kernel(sigma=sigma)
    return convolve(M, W)

def trap(f, ys, i, I):
    r = ys
    n=len(r)
    h = 1 / float(n)
    y0 = ys[0]
    intgr = 0.5 * h * (f(0, y0, i, I[0]) + f(1, y0, i, I[0]))
    for k in range(1, int(n)):
        intgr = intgr + h * f(k * h, y0, i, I[k])
    return intgr

def find_zero_pad(line, ymin=100, air=False, to_show=False):

    zind = find_zero_contour(line, ymin=ymin)

    if not air:
        length = 10 # padding to put at the edge of the new line
        newline = np.zeros(zind+int(length/2)) # initialize the new line removing the negative values
        end=line[zind-int(length/2)]/np.arange(1, length) # Pad the boundary where the phase goes negative
        newline[-(length-1):]=end
        newline[:-(int(length/2) )] = line[:zind]
        # newline = newline[zind:]
    else:
        newline = np.zeros_like(line)
        newline[zind:]=line[zind:]
        zind = find_zero_contour(np.abs(line), ymin=zind+25, delt=0.6)
        newline[zind:] = 0

    if to_show:
        plot_one_line(y=line, f_name="original", to_show=False)
        plot_one_line(y=newline, f_name='new line', to_show=True)

    return newline

def find_zero_contour(line, ymin=100, delt=0.001):
    try:    
        zind = np.where(line[ymin:] <=0+delt)[0][0]+ymin
    except:
        zind =len(line)
    return zind

def find_val_contour(line, ymin=100, delt=0.001, val=1):
    zind = np.where(line[ymin:] <=val+delt)[0][0]+ymin
    return zind

def generate_fft_mask(fftxcoords, fftycoords, F_ps, F_s, typ='box'):
    # create a mask
    if typ == 'box':
        if type(fftxcoords) == int:
            mask = gen_mask(F_ps.shape[0], F_ps.shape[1], (fftxcoords, int(F_ps.shape[1] / 2)), fftycoords)
        elif type(fftycoords) == int:
            mask = gen_mask(F_ps.shape[0], F_ps.shape[1], fftxcoords, (fftycoords, int(F_ps.shape[1] / 2)))
        else:
            mask = gen_mask(F_ps.shape[0], F_ps.shape[1], fftxcoords, fftycoords)
            Gsf = Gs
            s = 4
            kGs = sample_kernel(Gsf, s)
            mask = convolve(mask, kGs/np.sum(kGs))

            plot_one_thing(mask, "mask", colorbar=True)
    elif typ == 'gaussian':
            mask = gen_mask(F_ps.shape[0], F_ps.shape[1], fftxcoords, fftycoords, typ='gaussian', kxsize=50, kysize=20)

    return mask

def rotate_im(p, angle):
    pR = rotate(p, angle=angle)
    return pR


def get_ring_index(r, delt=10):
    f = np.zeros_like(r)
    l = len(f)
    lB = delt//2
    rB = lB+delt
    cent = int(l*2/3)
    f[cent-lB:cent+rB] = 1
    Gsf = Gs1d
    s = 4
    kGs = sample_kernel1d(Gsf, s)
    f = convolve(f, kGs / np.sum(kGs))

    return f

def calc_laser_phase(wl=1.0, fnc=None, r=None):
    phase = np.zeros_like(fnc)
    N = len(phase)
    deltr=r[1]-r[0]
    for n1 in range(1, N):
        sum=0
        for n2 in range(n1, N-1):
            sum=sum+fnc[n2]*(np.sqrt(r[n2+1]**2-r[n1]**2)-np.sqrt(r[n2]**2-r[n1]**2))

        phase[n1]=(2.0 * deltr / wl)*sum
    phase[0]=phase[1]
    return 100*phase


