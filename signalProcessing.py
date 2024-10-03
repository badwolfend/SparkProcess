from __future__ import print_function
import matplotlib as mpl
import Utils as ut
import numpy as np
import tracemalloc
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
outfile = TemporaryFile()

# Use LaTeX for rendering
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['font.size'] = 18
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

def main(fname='05255_t1_int', fname_ps=None, typ='box', size=600, coords = (1330, 850),
         fftycoords = (400, 500), fftxcoords=(475), not_horizontal=False, angle=0, downsamplef=1):
    f_name = fname
    f_name_h = f_name + ".jpg"
    if fname_ps is None:
        f_name_ps = f_name+"_preshot"
    else:
        f_name_ps = fname_ps+"_preshot"

    f_name_h_ps = f_name_ps + ".jpg"

    # Initially Perform Wavelet Analysis to clean #
    # Preshot #
    # Split Channels #
    b_o, g_o, r_o = ut.preProcess(f_name=f_name_h_ps, med_ksize=19)

    # Shot #
    # Split Channels #
    (b, g, r) = ut.preProcess(f_name=f_name_h, med_ksize=19)

    # Resize Images #
    ut.plot_one_thing(g, "original", colorbar=True, plot_3d=False, to_show=False)
    
    coordsx = coords[0]
    coordsy = coords[1]
    xcorrect = 0
    ycorrect = 0
    g = ut.resize_image(coords, size, g)
    g_o = ut.resize_image((coordsx+xcorrect, coordsy+ycorrect), size, g_o)
    ut.plot_one_thing(g, "shot", colorbar=True, plot_3d=False, to_show=False)
    ut.plot_one_thing(g_o, "ps", colorbar=True, plot_3d=False, to_show=True)

    # Downsample
    f=downsamplef
    fftxcoords = (int(fftxcoords[0]/f), int(fftxcoords[1]/f))
    fftycoords = (int(fftycoords[0]/f), int(fftycoords[1]/f))
    if f > 1:
        r_o = ut.downsample_2d(r_o, freq=f)
        g_o = ut.downsample_2d(g_o, freq=f)
        b_o = ut.downsample_2d(b_o, freq=f)
        r = ut.downsample_2d(r, freq=f)
        g = ut.downsample_2d(g, freq=f)
        b = ut.downsample_2d(b, freq=f)

    g_ps=g_o
    g_s =g

    # Now For Shot, Resize and Find Mask #
    b = []
    r = []
    b_o = []
    r_o = []

    if not_horizontal:
        g_s = np.rot90(g_s, 1)
        g_ps = np.rot90(g_ps, 1)
    # h_n, h_bin = ut.plot_hist_image(g_ps, 30, True)
    # ut.fit_to_dist(g_ps.flatten(), h_n, h_bin, 'gamma')

    # Pad with zeros to 2^N #
    ((F_s, F_ps), orig_size) = ut.pad_and_fft((g_s, g_ps))

    # create a mask
    # The mask is an array of 1s at the elements of the fft we want to keep.  The rest of the array is filled with zeros
    mask = ut.generate_fft_mask(fftxcoords=fftxcoords, fftycoords=fftycoords, F_ps=F_ps, F_s=F_s, typ=typ)

    # apply mask and inverse DFT #
    # First For Preshot then shot #
    ## pss and ss are the images back in the real space, but after filtering in the fourier domain. ##
    ## F_m_ps and F_m_s are the fourier domain images, but after the mask has been applied ##
    ((pss, F_m_ps), (ss, F_m_s)) = ut.apply_mask_ifft((F_ps, F_s), mask)
    (phase_ps, phase_s) = ut.resize_list_images(ut.compute_wrapped_phase((pss, ss)), orig_size)
    ut.plot_one_thing(phase_s, "phase_s", colorbar=True)

    # Unwrap Phase #
    ((ResS, RmaskS), (ResPS, RmaskPS)) = ut.gen_res_map((phase_s, phase_ps), int(10/f))
    ut.plot_one_thing(RmaskS, "residualS", colorbar=True)
    ut.plot_one_thing(RmaskPS, "residualPS", colorbar=True)
    ps, n_iters_s, err_s, maskedSigS = ut.modifiedCGSolver(phase_s, 1E-15, RmaskS, to_plot=True)
    pps, n_iters_ps, err_ps, maskedSigPS = ut.modifiedCGSolver(phase_ps, 1E-15, RmaskPS, to_plot=True)

    # Compute Difference #
    pdiff = ps-pps

    # Resize Images to original size #
    (pdiff, ps, F_ps, pps, ps) = ut.resize_list_images((pdiff, ps, F_ps, pps, ps), orig_size)
    pps = pps + abs(pps.min())
    ps = ps + abs(ps.min())

    pdiffR = ut.rotate_im(p=pdiff, angle=angle)
    ut.plot_one_thing(pdiffR, "pdiff", colorbar=True, plot_3d=False, to_show=False, vmax=(-20, 12))

    np.savez('Data/'+f_name+'_phasediff', pdiffR)
    np.savez('Data/'+f_name, pdiffR)

    # pdiffRLP = ut.gaussian_filter(pdiffR, sigma=6)
    # ut.plot_one_thing(pdiffRLP, "pdiffRLP", colorbar=True, plot_3d=False, vmax=(-20, 12))

    # Plot Everything #
    # ut.plot_one_thing(pdiff, "pdiff", colorbar=True, plot_3d=False)
    ut.plot_the_things(g_ps, phase_ps, pps, F_m_ps, F_ps, g_s, phase_s, ps, F_s, F_m_s, pdiff, f_name=f_name)


    # Plot Everything and save them#

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(phase_s, cmap="RdBu", vmin=0, vmax=6.5)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("./Plots/8_4_2023_shot1_phase_s.png", dpi=600)
    plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(pdiff, cmap="RdBu", vmin=-20, vmax=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("./Plots/8_4_2023_shot1_pdiff.png", dpi=600)
    plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14,14)
    im = ax1.imshow(np.log10(np.abs(F_s)+1), cmap="RdBu")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("./Plots/8_4_2023_shot1_F_s.png", dpi=600)
    plt.show()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    fig1.set_size_inches(14, 14)
    im = ax1.imshow(np.log10(np.abs(F_ps)+1), cmap="RdBu")
    plt.colorbar(im)
    plt.savefig("./Plots/8_4_2023_shot1_F_ps.png", dpi=600)
    plt.show()

    # ut.save_mat_to_im(pdiff, '8_4_2023_shot1_pdiff.tiff')

    # stopping the library
    tracemalloc.stop()
    return err_s, err_ps

if __name__ == '__main__':
    # # coords in (row, column).
    # e_s, e_ps = main(fname='Quantaray/8_4_2023/shot1', fname_ps="Quantaray/8_4_2023/shot1", size=(1400, 3200),
    #                 coords=(int(950), int(2000)), fftycoords=(int(1500), int(1550)), fftxcoords=(int(620), int(780)),
    #                 not_horizontal=True, typ='box', angle=0.0, downsamplef=1)
    # e_s, e_ps = main(fname='Quantaray/8_23_2024/shot0', fname_ps="Quantaray/8_23_2024/shot0", size=(936, 1732),
    #                 coords=(int(1935), int(2793)), fftycoords=(int(790), int(840)), fftxcoords=(int(300), int(625)),
    #                 not_horizontal=True, typ='box', angle=-10.0, downsamplef=1)
    # e_s, e_ps = main(fname='Quantaray/8_23_2024/00249_interferometer', fname_ps="Quantaray/8_23_2024/00015_interferometer", size=(936, 1732),
    #             coords=(int(2000), int(800)), fftycoords=(int(800), int(840)), fftxcoords=(int(300), int(625)),
    #             not_horizontal=True, typ='box', angle=0.0, downsamplef=1)

    # e_s, e_ps = main(fname='Quantaray/8_23_2024/00163_interferometer', fname_ps="Quantaray/8_23_2024/00015_interferometer", size=(936, 1732),
    #         coords=(int(2000), int(800)), fftycoords=(int(800), int(840)), fftxcoords=(int(300), int(625)),
    #         not_horizontal=True, typ='box', angle=0.0, downsamplef=1)

    #   e_s, e_ps = main(fname='Quantaray/8_23_2024/00118_interferometer', fname_ps="Quantaray/8_23_2024/00015_interferometer", size=(936, 1732),
    #         coords=(int(2000), int(2000+800)), fftycoords=(int(790), int(830)), fftxcoords=(int(470-150), int(470+150)),
    #         not_horizontal=True, typ='box', angle=0.0, downsamplef=1)

    # e_s, e_ps = main(fname='Quantaray/8_23_2024/00118_interferometer', fname_ps="Quantaray/8_23_2024/00015_interferometer", size=(936, 1732),
    # coords=(int(2000), int(800)), fftycoords=(int(790), int(830)), fftxcoords=(int(470-150), int(470+150)),
    # not_horizontal=True, typ='box', angle=0.0, downsamplef=1)

    e_s, e_ps = main(fname='Quantaray/8_23_2024/00249_interferometer', fname_ps="Quantaray/8_23_2024/00015_interferometer", size=(936, 1732),
    coords=(int(1930), int(800)), fftycoords=(int(790), int(830)), fftxcoords=(int(470-150), int(470+150)),
    not_horizontal=True, typ='box', angle=0.0, downsamplef=1)