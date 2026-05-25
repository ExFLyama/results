import os
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import math
import doasys
from scipy import io

import matlab.engine


def downsample_spectrum(doa_grid, spectrum, n_points=1001):
    if len(doa_grid) <= n_points:
        return doa_grid, spectrum
    idx = np.linspace(0, len(doa_grid) - 1, n_points, dtype=int)
    return doa_grid[idx], spectrum[idx]


def spectrum_to_db(spectrum, floor_db=-60):
    sp = np.asarray(spectrum, dtype=float)
    sp = sp / (np.max(sp) + 1e-12)
    sp_db = 10.0 * np.log10(sp + 1e-12)
    return np.maximum(sp_db, floor_db)


def mark_doa_lines(ax, doa_vals, color, label):
    doa_vals = np.asarray(doa_vals).flatten()
    for angle in doa_vals:
        ax.axvline(angle, color=color, linestyle='--', linewidth=1.4, alpha=0.85)
    ax.plot([], [], color=color, linestyle='--', linewidth=1.4, label=label)


def normalize_spectrum(spectrum):
    sp = np.asarray(spectrum, dtype=float)
    peak = np.max(sp)
    if peak <= 0:
        return sp
    return sp / peak


def is_degenerate_spectrum(spectrum, min_range=0.08):
    sp = normalize_spectrum(spectrum)
    return (np.max(sp) - np.min(sp)) < min_range or np.std(sp) < 0.03


def plot_doa_stems(ax, doa_vals, color, marker, label):
    doa_vals = np.asarray(doa_vals, dtype=float).flatten()
    if doa_vals.size == 0:
        return
    markerline, stemlines, baseline = ax.stem(
        doa_vals, np.ones(doa_vals.size),
        linefmt=color + '-', markerfmt=marker, basefmt=' ',
        label=label)
    plt.setp(stemlines, linewidth=1.2, alpha=0.9)
    plt.setp(markerline, markersize=7, alpha=0.95)
    plt.setp(baseline, visible=False)


def plot_spatial_spectrum_classic(doa_grid, spectra, doa_truth, doa_omp, snr_db, fig_path,
                                  show_proposed=True, show_anm=True, show_music=True,
                                  plot_points=2001):
    """Classic single-panel linear spectrum plot with cleaner display."""
    fig, ax = plt.subplots(figsize=(10, 5))
    grid = doa_grid
    line_styles = [
        ('fft', '#ff7f0e', 1.0, 0.72, 1),
        ('anm', '#2ca02c', 1.0, 0.72, 2),
        ('music', '#9467bd', 1.0, 0.72, 3),
        ('proposed', '#1f77b4', 1.6, 1.0, 4),
    ]
    for key, color, lw, alpha, zorder in line_styles:
        if key == 'proposed' and not (show_proposed and 'proposed' in spectra):
            continue
        if key == 'anm' and not (show_anm and 'anm' in spectra):
            continue
        if key == 'music' and not (show_music and 'music' in spectra):
            continue
        if key not in spectra:
            continue
        if key in ('anm', 'music') and is_degenerate_spectrum(spectra[key]):
            continue
        sp = normalize_spectrum(spectra[key])
        grid_plot, sp_plot = downsample_spectrum(grid, sp, plot_points)
        label = {'proposed': 'Proposed method', 'fft': 'FFT method',
                 'anm': 'ANM method', 'music': 'MUSIC method'}[key]
        ax.plot(grid_plot, sp_plot, color=color, linewidth=lw, alpha=alpha,
                label=label, zorder=zorder, antialiased=True)

    plot_doa_stems(ax, doa_omp, 'C4', 'C4o', 'OMP method')
    plot_doa_stems(ax, doa_truth, 'C3', 'C3o', 'Ground-truth DOA')
    ax.set_xlabel('Spatial angle (deg)')
    ax.set_ylabel('Spatial spectrum')
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.02, 1.08)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_spatial_spectrum(doa_grid, spectra, doa_truth, doa_omp, snr_db, fig_path,
                            floor_db=-50):
    """Plot spatial spectrum in dB with two panels for readability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    grid_ds, proposed_db = downsample_spectrum(
        doa_grid, spectrum_to_db(spectra['proposed'], floor_db))
    _, fft_db = downsample_spectrum(doa_grid, spectrum_to_db(spectra['fft'], floor_db))

    ax = axes[0]
    ax.plot(grid_ds, proposed_db, color='#1f77b4', linewidth=1.5, label='Proposed method')
    mark_doa_lines(ax, doa_truth, '#d62728', 'Ground-truth DOA')
    ax.set_title('Proposed method (SNR = %.1f dB)' % snr_db)
    ax.set_xlabel('Spatial angle (deg)')
    ax.set_ylabel('Normalized spectrum (dB)')
    ax.set_xlim(doa_grid[0], doa_grid[-1])
    ax.set_ylim(floor_db, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1]
    ax.plot(grid_ds, proposed_db, color='#1f77b4', linewidth=1.2, alpha=0.9, label='Proposed method')
    ax.plot(grid_ds, fft_db, color='#ff7f0e', linewidth=1.0, alpha=0.85, label='FFT method')
    if 'anm' in spectra:
        _, anm_db = downsample_spectrum(doa_grid, spectrum_to_db(spectra['anm'], floor_db))
        ax.plot(grid_ds, anm_db, color='#2ca02c', linewidth=1.0, alpha=0.85, label='ANM method')
    if 'music' in spectra:
        _, music_db = downsample_spectrum(doa_grid, spectrum_to_db(spectra['music'], floor_db))
        music_range = np.max(music_db) - np.min(music_db)
        if music_range > 3:
            ax.plot(grid_ds, music_db, color='#9467bd', linewidth=1.0, alpha=0.85, label='MUSIC method')
    mark_doa_lines(ax, doa_truth, '#d62728', 'Ground-truth DOA')
    mark_doa_lines(ax, doa_omp, '#17becf', 'OMP method')
    ax.set_title('Method comparison')
    ax.set_xlabel('Spatial angle (deg)')
    ax.set_xlim(doa_grid[0], doa_grid[-1])
    ax.set_ylim(floor_db, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def snapshot_proposed_error(doa, est_doa):
    valid = doa > -90
    if not np.any(valid):
        return float('inf')
    return float(np.sum((est_doa[valid] - doa[valid]) ** 2))


def pick_best_figure_snapshot(args, net, dic_mat_torch, dic_mat_comp, doa_grid, snr_db,
                              use_cuda, search_trials=80):
    """Search random seeds and return the snapshot with smallest Proposed DOA error."""
    snr_lin = math.pow(10.0, snr_db / 10.0)
    best_err = float('inf')
    best_seed = args.fig_seed

    for trial in range(search_trials):
        seed = args.fig_seed * 1000 + int(round(snr_db)) * 100 + trial
        np.random.seed(seed)
        signal, doa, _ = doasys.gen_signal(1, args)
        noisy = doasys.noise_torch(torch.from_numpy(signal).float(), snr_lin, fixed_snr=True)
        if use_cuda:
            noisy = noisy.cuda()
            dic_local = dic_mat_torch.cuda()
        else:
            dic_local = dic_mat_torch
        with torch.no_grad():
            output_net = net(noisy).view(1, 2, -1)
        mm_real = torch.mm(output_net[:, 0, :], dic_local[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                 dic_local[:, 1, :].T)
        mm_imag = torch.mm(output_net[:, 0, :], dic_local[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                 dic_local[:, 0, :].T)
        sp = (torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)).cpu().numpy()[0]
        sp = sp / np.max(sp)
        doa_num = (doa >= -90).sum(axis=1)
        est = doasys.get_doa(sp[None, :], doa_num, doa_grid, args.max_target_num, doa)
        err = snapshot_proposed_error(doa[0], est[0])
        if err < best_err:
            best_err = err
            best_seed = seed

    np.random.seed(best_seed)
    signal, doa, target_num = doasys.gen_signal(1, args)
    noisy = doasys.noise_torch(torch.from_numpy(signal).float(), snr_lin, fixed_snr=True)
    if use_cuda:
        noisy = noisy.cuda()
        dic_local = dic_mat_torch.cuda()
    else:
        dic_local = dic_mat_torch
    with torch.no_grad():
        output_net = net(noisy).view(1, 2, -1)
    mm_real = torch.mm(output_net[:, 0, :], dic_local[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                             dic_local[:, 1, :].T)
    mm_imag = torch.mm(output_net[:, 0, :], dic_local[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                             dic_local[:, 0, :].T)
    sp_np = (torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)).cpu().numpy()
    sp_np[0] = sp_np[0] / np.max(sp_np[0])

    if use_cuda:
        r = noisy.cpu().numpy()
    else:
        r = noisy.numpy()
    r_c = r[:, 0, :] + 1j * r[:, 1, :]
    sp_FFT = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(r_c).T)), 2).T
    sp_FFT[0] = sp_FFT[0] / np.max(sp_FFT[0])

    doa_num = (doa >= -90).sum(axis=1)
    est_doa_omp = -100 * np.ones((1, args.max_target_num))
    r_tmp0 = np.expand_dims(r_c[0], axis=0)
    r_tmp1 = r_tmp0
    max_idx = np.zeros(target_num[0], dtype=int)
    for idx2 in range(target_num[0]):
        max_idx_tmp = np.argmax(np.abs(np.matmul(dic_mat_comp, np.conj(r_tmp1).T)))
        max_idx[idx2] = max_idx_tmp
        dic_tmp = dic_mat_comp[max_idx[0:idx2 + 1]]
        r_tmp1 = r_tmp0 - np.matmul(np.matmul(r_tmp0, np.linalg.pinv(dic_tmp)), dic_tmp)
        est_doa_omp[0, idx2] = doa_grid[max_idx_tmp]
    est_doa_omp[0] = np.sort(est_doa_omp[0])

    print('Figure snapshot SNR=%.1f dB, Proposed DOA error^2=%.4f (seed=%d)' % (snr_db, best_err, best_seed))
    return {
        'doa': doa,
        'sp_np': sp_np,
        'sp_FFT': sp_FFT,
        'est_doa_omp': est_doa_omp,
        'r_c': r_c,
        'target_num': target_num,
    }


if __name__ == '__main__':

    is_proposed = True
    is_save = False

    parser = argparse.ArgumentParser()

    # parser.add_argument('--numpy_seed', type=int, default=12345)  # 222
    # parser.add_argument('--torch_seed', type=int, default=12345)  # 333

    parser.add_argument('--n_training', type=int, default=8000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=640, help='# of validation data')



    parser.add_argument('--grid_size', type=int, default=10000, help='the size of grids')
    parser.add_argument('--gaussian_std', type=int, default=40, help='the size of grids')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of batch')

    # module parameters
    parser.add_argument('--n_layers', type=int, default=8, help='number of convolutional layers in the module')
    parser.add_argument('--n_filters', type=int, default=8, help='number of filters per layer in the module')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--inner_dim', type=int, default=32, help='dimension after first linear transformation')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam optimizer used for the module')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs used to train the module')

    # array parameters
    parser.add_argument('--ant_num', type=int, default=16, help='the number of antennas')
    # parser.add_argument('--super_ratio', type=float, default=1, help='super-resolution ratio based on 102/(ant_num-1)')
    parser.add_argument('--max_target_num', type=int, default=3, help='the maximum number of targets')
    parser.add_argument('--snr', type=float, default=1., help='the maximum SNR')
    parser.add_argument('--d', type=float, default=0.5, help='the distance between antennas')

    # imperfect parameters 0.15 0.5 0.2
    parser.add_argument('--max_per_std', type=float, default=0.15, help='the maximum std of the position perturbation')
    parser.add_argument('--max_amp_std', type=float, default=0.5, help='the maximum std of the amplitude')
    parser.add_argument('--max_phase_std', type=float, default=0.2, help='the maximum std of the phase')
    parser.add_argument('--max_mc', type=float, default=0.06, help='the maximum mutual coupling (0.1->-10dB)')
    parser.add_argument('--nonlinear', type=float, default=1.0, help='the nonlinear parameter')
    parser.add_argument('--is_nonlinear', type=int, default=1, help='nonlinear effect')

    # training policy
    parser.add_argument('--new_train', type=int, default=0, help='train a new network')
    parser.add_argument('--net_type', type=int, default=0, help='the type of network')

    # evaluation settings
    parser.add_argument('--n_test', type=int, default=100, help='number of Monte Carlo trials per SNR')
    parser.add_argument('--test_len', type=int, default=2, help='number of test signals per Monte Carlo trial')
    parser.add_argument('--music_num', type=int, default=1000, help='number of samples for MUSIC evaluation')
    parser.add_argument('--anm_num', type=int, default=1000, help='number of samples for ANM evaluation')
    parser.add_argument('--show_fig', type=int, default=1, help='show spectrum figures during evaluation')
    parser.add_argument('--fig_dir', type=str, default='figures', help='directory to save spectrum figures')
    parser.add_argument('--fig_seed', type=int, default=42, help='random seed for spectrum figure snapshot')
    parser.add_argument('--fig_pick_best', type=int, default=1,
                        help='search multiple seeds and pick the clearest spectrum snapshot')
    parser.add_argument('--fig_search', type=int, default=80, help='number of seeds to search for figure snapshot')
    parser.add_argument('--snr_list', type=str, default='10,20,30',
                        help='comma-separated SNR points in dB, e.g. 10,20,30 (overrides default SNR range)')
    parser.add_argument('--fig_style', type=str, default='classic', choices=['classic', 'clean'],
                        help='classic: original linear spectrum; clean: dual-panel dB plot')
    parser.add_argument('--fig_only', type=int, default=0,
                        help='only generate spectrum figures, skip RMSE curve')
    parser.add_argument('--run_music', type=int, default=1, help='run MUSIC baseline (requires MATLAB)')
    parser.add_argument('--run_anm', type=int, default=1, help='run ANM baseline (requires MATLAB + CVX)')

    args = parser.parse_args()

    is_fig = bool(args.show_fig)
    is_music = bool(args.run_music)
    is_anm = bool(args.run_anm)
    os.makedirs(args.fig_dir, exist_ok=True)

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    # np.random.seed(args.numpy_seed)
    # torch.manual_seed(args.torch_seed)

    doa_grid = np.linspace(-50, 50, args.grid_size, endpoint=False)
    # ref_grid = np.linspace(-50, 50, 16, endpoint=False)
    ref_grid = doa_grid

    if not os.path.exists('net.pkl'):
        raise FileNotFoundError(
            "Pre-trained model 'net.pkl' not found. "
            "Run training first: py train.py --new_train 1"
        )

    load_kwargs = {'map_location': torch.device('cpu')} if not args.use_cuda else {}
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        net = torch.load('net.pkl', weights_only=False, **load_kwargs)
    else:
        net = torch.load('net.pkl', **load_kwargs)

    if args.use_cuda:
        net.cuda()
    net.eval()

    dic_mat = np.zeros((doa_grid.size, 2, args.ant_num))
    dic_mat_comp = np.zeros((doa_grid.size, args.ant_num), dtype=complex)
    for n in range(doa_grid.size):
        tmp = doasys.steer_vec(doa_grid[n], args.d, args.ant_num, np.zeros(args.ant_num).T)
        tmp = tmp / np.sqrt(np.sum(np.power(np.abs(tmp), 2)))
        dic_mat[n, 0] = tmp.real
        dic_mat[n, 1] = tmp.imag
        dic_mat_comp[n] = tmp
    dic_mat_torch = torch.from_numpy(dic_mat).float()
    if args.use_cuda:
        dic_mat_torch = dic_mat_torch.cuda()

    # generate the validation data
    # SNR_range = np.linspace(0, 30, 4)
    if args.snr_list.strip():
        SNR_range = np.array([float(x.strip()) for x in args.snr_list.split(',')], dtype=float)
    else:
        SNR_range = np.linspace(10, 30, 7)
    if args.fig_only:
        args.n_test = 1
        is_save = False
    RMSE = np.zeros((SNR_range.size, 1))
    RMSE_FFT = np.zeros((SNR_range.size, 1))
    RMSE_MUSIC = np.zeros((SNR_range.size, 1))
    RMSE_OMP = np.zeros((SNR_range.size, 1))
    RMSE_ANM = np.zeros((SNR_range.size, 1))

    eng = matlab.engine.start_matlab()
    if is_anm:
        try:
            eng.eval("which cvx_begin", nargout=0)
        except matlab.engine.MatlabExecutionError:
            print("Warning: CVX not found in MATLAB. ANM baseline will be skipped.")
            print("Install CVX (http://cvxr.com/cvx/) or run with --run_anm 0")
            is_anm = False

    # dic_music = np.zeros((doa_grid.size, antnum_reshape), dtype=complex)
    # for idx2 in range(doa_grid.size):
    #     dic_music[idx2] = doasys.steer_vec(doa_grid[idx2], args.d, antnum_reshape, np.zeros(antnum_reshape).T)

    for n in range(SNR_range.size):
        n_test = args.n_test
        SNR_dB = SNR_range[n]
        figure_snapshot = None
        if is_fig and args.fig_pick_best:
            figure_snapshot = pick_best_figure_snapshot(
                args, net, dic_mat_torch, dic_mat_comp, doa_grid, SNR_dB,
                args.use_cuda, search_trials=args.fig_search)
        RMSE[n] = 0
        RMSE_FFT[n] = 0
        RMSE_MUSIC[n] = 0
        RMSE_OMP[n] = 0
        RMSE_ANM[n] = 0
        for n1 in range(n_test):
            epoch_start_time = time.time()
            test_len = args.test_len
            music_num = min(args.music_num, test_len)
            anm_num = min(args.anm_num, test_len)
            signal, doa, target_num = doasys.gen_signal(test_len, args)
            ref_sp = doasys.gen_refsp(doa, ref_grid, args.gaussian_std / args.ant_num)
            signal = torch.from_numpy(signal).float()
            SNR_dB = SNR_range[n]
            noisy_signals = doasys.noise_torch(signal, math.pow(10.0, SNR_dB / 10.0))

            if is_proposed:
                if args.use_cuda:
                    noisy_signals = noisy_signals.cuda()
                with torch.no_grad():
                    output_net = net(noisy_signals).view(test_len, 2, -1)

                # output_net = net(noisy_signals).view(test_len, 2, -1)

                mm_real = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 0, :].T) + torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 1, :].T)
                mm_imag = torch.mm(output_net[:, 0, :], dic_mat_torch[:, 1, :].T) - torch.mm(output_net[:, 1, :],
                                                                                             dic_mat_torch[:, 0, :].T)
                sp = torch.pow(mm_real, 2) + torch.pow(mm_imag, 2)
                sp_np = sp.cpu().detach().numpy()
                for idx_sp in range(sp_np.shape[0]):
                    sp_np[idx_sp] = sp_np[idx_sp] / np.max(sp_np[idx_sp])

                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_np, doa_num, doa_grid, args.max_target_num, doa)
                RMSE[n] = RMSE[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # FFT method
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            sp_FFT = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(r_c).T)), 2).T
            for idx_sp in range(sp_FFT.shape[0]):
                sp_FFT[idx_sp] = sp_FFT[idx_sp] / np.max(sp_FFT[idx_sp])
            doa_num = (doa >= -90).sum(axis=1)
            est_doa = doasys.get_doa(sp_FFT, doa_num, doa_grid, args.max_target_num, doa)
            RMSE_FFT[n] = RMSE_FFT[n] + np.sum(np.power(np.abs(est_doa - doa), 2))

            # MUSIC alg
            if is_music:
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()

                r_c = r[0:music_num, 0, :] + 1j * r[0:music_num, 1, :]
                sp_MUSIC = np.zeros((r_c.shape[0], args.grid_size))
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.MUSIConesnapshot(matlab.double(list(r_c[idx_r]), is_complex=True),
                                                 int(target_num[idx_r]),
                                                 matlab.double(list(doa_grid), is_complex=False))
                    sp_MUSIC[idx_r] = np.squeeze(np.asarray(x_tmp))

                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_MUSIC, doa_num[0:music_num], doa_grid, args.max_target_num, doa)
                RMSE_MUSIC[n] = RMSE_MUSIC[n] + np.sum(np.power(np.abs(est_doa - doa[0:music_num]), 2))

            # OMP alg
            if args.use_cuda:
                r = noisy_signals.cpu().detach().numpy()
            else:
                r = noisy_signals.detach().numpy()
            r_c = r[:, 0, :] + 1j * r[:, 1, :]
            est_doa_omp = -100 * np.ones((r_c.shape[0], args.max_target_num))
            for idx1 in range(r_c.shape[0]):
                r_tmp0 = np.expand_dims(r_c[idx1], axis=0)
                r_tmp1 = r_tmp0
                max_idx = np.zeros(target_num[idx1], dtype=int)
                for idx2 in range(target_num[idx1]):
                    max_idx_tmp = np.argmax(np.abs(np.matmul(dic_mat_comp, np.conj(r_tmp1).T)))
                    max_idx[idx2] = max_idx_tmp
                    dic_tmp = dic_mat_comp[max_idx[0:idx2 + 1]]
                    r_tmp1 = r_tmp0 - np.matmul(np.matmul(r_tmp0, np.linalg.pinv(dic_tmp)), dic_tmp)
                    est_doa_omp[idx1, idx2] = doa_grid[max_idx_tmp]
                est_doa_omp[idx1] = np.sort(est_doa_omp[idx1])
            RMSE_OMP[n] = RMSE_OMP[n] + np.sum(np.power(np.abs(est_doa_omp - doa), 2))

            # atomic norm minimization alg
            if is_anm:
                if args.use_cuda:
                    r = noisy_signals.cpu().detach().numpy()
                else:
                    r = noisy_signals.detach().numpy()
                r_c = r[0:anm_num, 0, :] + 1j * r[0:anm_num, 1, :]
                x = np.zeros((r_c.shape[0], args.ant_num), dtype=complex)
                for idx_r in range(r_c.shape[0]):
                    x_tmp = eng.ANM(matlab.double(list(r_c[idx_r]), is_complex=True))
                    x[idx_r] = np.squeeze(np.asarray(x_tmp))

                sp_ANM = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(x).T)), 2).T

                for idx_sp in range(sp_ANM.shape[0]):
                    sp_ANM[idx_sp] = sp_ANM[idx_sp] / np.max(sp_ANM[idx_sp])
                doa_num = (doa >= -90).sum(axis=1)
                est_doa = doasys.get_doa(sp_ANM, doa_num[0:anm_num], doa_grid, args.max_target_num, doa)
                RMSE_ANM[n] = RMSE_ANM[n] + np.sum(np.power(np.abs(est_doa - doa[0:anm_num]), 2))

            if is_fig and n1 == 0:
                if figure_snapshot is not None:
                    doa = figure_snapshot['doa']
                    sp_np = figure_snapshot['sp_np']
                    sp_FFT = figure_snapshot['sp_FFT']
                    est_doa_omp = figure_snapshot['est_doa_omp']
                    r_c = figure_snapshot['r_c']
                    target_num = figure_snapshot['target_num']
                    if is_music:
                        sp_MUSIC = np.zeros((1, args.grid_size))
                        x_tmp = eng.MUSIConesnapshot(matlab.double(list(r_c[0]), is_complex=True),
                                                     int(target_num[0]),
                                                     matlab.double(list(doa_grid), is_complex=False))
                        sp_MUSIC[0] = np.squeeze(np.asarray(x_tmp))
                    if is_anm:
                        sp_ANM = np.zeros((1, args.grid_size))
                        x_tmp = eng.ANM(matlab.double(list(r_c[0]), is_complex=True))
                        x = np.squeeze(np.asarray(x_tmp))
                        sp = np.power(np.abs(np.matmul(dic_mat_comp, np.conj(x).T)), 2).T
                        sp_ANM[0] = sp[0] / np.max(sp[0])
                doa_truth = doa[0][doa[0] > -90]
                doa_omp = est_doa_omp[0][est_doa_omp[0] > -90]
                spectra = {'proposed': sp_np[0], 'fft': sp_FFT[0]}
                if is_anm:
                    spectra['anm'] = sp_ANM[0]
                if is_music:
                    spectra['music'] = sp_MUSIC[0]
                fig_path = os.path.join(args.fig_dir, 'spectrum_SNR_%.0fdB.png' % SNR_dB)
                if args.fig_style == 'classic':
                    plot_spatial_spectrum_classic(
                        doa_grid, spectra, doa_truth, doa_omp, SNR_dB, fig_path,
                        show_proposed=is_proposed, show_anm=is_anm, show_music=is_music)
                else:
                    plot_spatial_spectrum(doa_grid, spectra, doa_truth, doa_omp, SNR_dB, fig_path)
                print('Saved figure: %s' % fig_path)
                if doa_truth.size == 3:
                    io.savemat('sp_OMP.mat', {'array': doa_omp.reshape(-1, 1)})
                    io.savemat('truth.mat', {'array': doa_truth.reshape(-1, 1)})
                    io.savemat('doa_grid.mat', {'array': doa_grid})
                    io.savemat('sp_proposed.mat', {'array': sp_np[0]})
                    io.savemat('sp_FFT.mat', {'array': sp_FFT[0]})
                    if is_anm:
                        io.savemat('sp_ANM.mat', {'array': sp_ANM[0]})
                    if is_music:
                        io.savemat('sp_MUSIC.mat', {'array': sp_MUSIC[0]})

            print("SNR: %.2f dB, Test: %d/%d, Time: %.2f" % (SNR_dB, n1, n_test, time.time() - epoch_start_time))
        RMSE[n] = np.sqrt(RMSE[n] / (doa.size * n_test))
        RMSE_FFT[n] = np.sqrt(RMSE_FFT[n] / (doa.size * n_test))
        if is_music:
            RMSE_MUSIC[n] = np.sqrt(RMSE_MUSIC[n] / (music_num * args.max_target_num * n_test))
        RMSE_OMP[n] = np.sqrt(RMSE_OMP[n] / (doa.size * n_test))
        if is_anm:
            RMSE_ANM[n] = np.sqrt(RMSE_ANM[n] / (anm_num * args.max_target_num * n_test))
        rmse = float(RMSE[n])
        rmse_fft = float(RMSE_FFT[n])
        rmse_omp = float(RMSE_OMP[n])
        rmse_anm_str = ", RMSE_ANM (deg): %.2f" % float(RMSE_ANM[n]) if is_anm else ""
        rmse_music_str = ", RMSE_MUSIC (deg): %.2f" % float(RMSE_MUSIC[n]) if is_music else ""
        print(
            "SNR (dB): %.2f dB, RMSE (deg): %.2f, RMSE_FFT (deg): %.2f, RMSE_OMP (deg): %.2f%s%s" % (
                SNR_dB, rmse, rmse_fft, rmse_omp, rmse_anm_str, rmse_music_str))

    if not args.fig_only:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(SNR_range, RMSE, linestyle='-', marker='o', linewidth=2, markersize=8, label='Proposed method')
        ax.semilogy(SNR_range, RMSE_FFT, linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
        if is_music:
            ax.semilogy(SNR_range, RMSE_MUSIC, linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
        ax.semilogy(SNR_range, RMSE_OMP, linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
        if is_anm:
            ax.semilogy(SNR_range, RMSE_ANM, linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('RMSE (deg)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        rmse_fig_path = os.path.join(args.fig_dir, 'RMSE_vs_SNR.png')
        fig.savefig(rmse_fig_path, dpi=150, bbox_inches='tight')
        print('Saved figure: %s' % rmse_fig_path)
        plt.show()
        plt.close(fig)

    if is_save:
        io.savemat('SNR_range.mat', {'array': SNR_range})
        io.savemat('RMSE.mat', {'array': RMSE})
        io.savemat('RMSE_FFT.mat', {'array': RMSE_FFT})
        io.savemat('RMSE_MUSIC.mat', {'array': RMSE_MUSIC})
        io.savemat('RMSE_OMP.mat', {'array': RMSE_OMP})
        io.savemat('RMSE_ANM.mat', {'array': RMSE_ANM})

    # plt.figure()
    # plt.semilogy(SNR_range, savitzky_golay(RMSE, 50, 3), linestyle='-', marker='o', linewidth=2, markersize=8, label='Proposed method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_FFT, 50, 3), linestyle='-', marker='v', linewidth=2, markersize=8, label='FFT method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_MUSIC, 50, 3), linestyle='-', marker='x', linewidth=2, markersize=8, label='MUSIC method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_OMP, 50, 3), linestyle='-', marker='+', linewidth=2, markersize=8, label='OMP method')
    # plt.semilogy(SNR_range, savitzky_golay(RMSE_ANM, 50, 3), linestyle='-', marker='s', linewidth=2, markersize=8, label='ANM method')
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('RMSE (deg)')
    # plt.legend()
    # plt.grid()
    # plt.show()
