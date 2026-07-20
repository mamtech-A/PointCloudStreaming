# -*- coding: utf-8 -*-
"""Generate the IST'2026 paper result charts (Fig. 3-7) in IEEE single-column style.

Outputs PNG (300 dpi) + vector PDF per chart under docs/figures/Fig{3..7}/.
Data sources: models/dqn_sweep_results.json, models/fixed_arm_baseline.json,
logs/dqn/results.csv, bandwidth_5g traces, round-4 trial history JSONs.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r"C:\Users\Mamtech\PointCloudStreaming"
FIGS = os.path.join(ROOT, "docs", "figures")

# ---------------------------------------------------------------- IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Okabe-Ito palette (grayscale-safe, matches Fig. 1/2 colors)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_PURPLE = '#6F4C9B'
C_PINK = '#CC79A7'
C_GREY = '#6B7280'
C_SKY = '#56B4E9'
C_VERM = '#D55E00'

COL_W = 3.5   # inches, IEEE single column

TIER_BITRATE = {0: 102.4, 1: 59.6, 2: 35.5, 3: 14.1, 4: 3.4, 5: 0.9}
TIER_NAME = {0: 'r06', 1: 'r05', 2: 'r04', 3: 'r03', 4: 'r02', 5: 'r01'}

TRACES = [
    ('driving_B_2019.12.14_10.16.30.csv', 'Driving A\n(blackout)'),
    ('driving_B_2019.12.16_07.22.43.csv', 'Driving B'),
    ('driving_B_2020.02.27_20.35.57.csv', 'Driving C'),
    ('static_B_2020.01.16_10.43.34.csv', 'Static'),
]

sweep = json.load(open(os.path.join(ROOT, 'models', 'dqn_sweep_results.json')))
arms = json.load(open(os.path.join(ROOT, 'models', 'fixed_arm_baseline.json')))['results']
winner = sweep['winner']['metrics']


def save(fig, sub, name):
    d = os.path.join(FIGS, sub)
    os.makedirs(d, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(d, f'{name}.{ext}'))
    plt.close(fig)
    print(f'saved {sub}/{name}.png|.pdf')


# ================================================================ Fig. 3
def fig3_per_trace_bars():
    dqn = [winner['qoe_quality_per_trace'][t] for t, _ in TRACES]
    r03 = [arms['arm_3']['per_trace'][t]['qoe_quality'] for t, _ in TRACES]
    r04 = [arms['arm_2']['per_trace'][t]['qoe_quality'] for t, _ in TRACES]

    x = np.arange(len(TRACES))
    w = 0.26
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    b1 = ax.bar(x - w, dqn, w, label='DQN (proposed)', color=C_PURPLE, edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x, r03, w, label='Always r03 (best fixed)', color=C_BLUE, edgecolor='black', linewidth=0.4, hatch='//')
    b3 = ax.bar(x + w, r04, w, label='Always r04', color=C_GREY, edgecolor='black', linewidth=0.4, hatch='..')
    for bars in (b1, b2, b3):
        for b in bars:
            v = b.get_height()
            off = (0, 1.5) if v >= 0 else (0, -8)
            ax.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                        textcoords='offset points', xytext=off,
                        ha='center', fontsize=6.2)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in TRACES])
    ax.set_ylabel('QoE')
    ax.set_ylim(-18, 100)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'Fig3', 'per_trace_qoe')


# ================================================================ Fig. 4
def fig4_timeseries():
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'logs', 'dqn', 'results.csv'))))
    # one entry per segment: start time (cumulative at previous segment end) + tier
    seg_t, seg_rate = [], []
    prev_end = 0.0
    seen = set()
    for r in rows:
        sid = int(r['segment_id'])
        if sid not in seen:
            seen.add(sid)
            seg_t.append(prev_end)
            seg_rate.append(TIER_BITRATE[int(r['rep_id'])])
        prev_end = float(r['cumulative_time_s'])
    seg_t.append(prev_end)          # close the last step
    seg_rate.append(seg_rate[-1])
    t_end = prev_end

    # capacity from the trace CSV (1 sample/s), same wall-clock axis
    tr = list(csv.DictReader(open(glob.glob(os.path.join(
        ROOT, 'bandwidth_5g', '**', 'static_B_2020.01.16_10.43.34.csv'), recursive=True)[0])))
    cap_t = np.arange(len(tr), dtype=float)
    cap = np.array([float(r['DL_bitrate']) for r in tr]) / 1000.0  # kbps -> Mbps
    m = cap_t <= t_end + 1
    cap_t, cap = cap_t[m], cap[m]

    # buffer + stalls per frame
    buf_t = [float(r['cumulative_time_s']) for r in rows]
    buf = [float(r['buffer_level_s']) for r in rows]
    stall_spans = [(float(r['cumulative_time_s']) - float(r['stall_duration_s']),
                    float(r['cumulative_time_s']))
                   for r in rows if r['stall'] == 'yes' and float(r['stall_duration_s']) > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True,
                                   gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.12})
    ax1.fill_between(cap_t, cap, color=C_SKY, alpha=0.35, step='post', linewidth=0)
    ax1.step(cap_t, cap, where='post', color=C_BLUE, linewidth=0.9, label='Link capacity')
    ax1.step(seg_t, seg_rate, where='post', color=C_PURPLE, linewidth=1.4, label='Selected tier bitrate')
    ax1.set_ylabel('Rate (Mbit/s)')
    ax1.set_ylim(bottom=0)
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', frameon=False)

    ax2.plot(buf_t, buf, color=C_GREEN, linewidth=1.0, label='Buffer level')
    for a, b in stall_spans:
        ax2.axvspan(a, b, color=C_VERM, alpha=0.3, linewidth=0)
    if stall_spans:
        ax2.axvspan(0, 0, color=C_VERM, alpha=0.3, label='Stall')
    ax2.set_ylabel('Buffer (s)')
    ax2.set_xlabel('Download wall-clock time (s)')
    ax2.set_ylim(0, 5.3)
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc='lower right', frameon=False)
    ax2.set_xlim(0, t_end + 0.3)
    save(fig, 'Fig4', 'policy_timeseries')


# ================================================================ Fig. 5
def fig5_segment_curve():
    S = [1, 5, 8, 10, 15, 30]
    qoe = [-9.8, 69.4, 65.4, 53.9, 54.9, 7.5]
    stall = [19.1, 2.9, 4.3, 7.5, 8.2, 20.4]

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axvspan(5, 8, color=C_GREEN, alpha=0.12, linewidth=0)
    l1, = ax.plot(S, qoe, marker='o', markersize=4, color=C_PURPLE, label='QoE (left)')
    ax.set_xlabel('Segment size S (frames per request)')
    ax.set_ylabel('QoE')
    ax.set_xticks(S)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    l2, = ax2.plot(S, stall, marker='s', markersize=3.5, linestyle='--',
                   color=C_VERM, label='Stall (right)')
    ax2.set_ylabel('Stall (s)')
    ax2.set_ylim(bottom=0)

    ax.annotate('sweet spot', xy=(6.5, 70), xytext=(12, 78),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', linewidth=0.6))
    ax.legend(handles=[l1, l2], loc='center right', frameon=False)
    save(fig, 'Fig5', 'segment_size_curve')


# ================================================================ Fig. 6
def fig6_cdf():
    series = [('DQN (proposed)',
               [winner['qoe_quality_per_trace'][t] for t, _ in TRACES],
               C_PURPLE, '-', 'o')]
    style = {
        'arm_0': ('Always r06', C_VERM, ':', 'v'),
        'arm_1': ('Always r05', C_ORANGE, '--', '^'),
        'arm_2': ('Always r04', C_GREY, '-.', 's'),
        'arm_3': ('Always r03', C_BLUE, '--', 'D'),
        'arm_4': ('Always r02', C_GREEN, ':', 'x'),
        'arm_5': ('Always r01', C_PINK, '-.', '+'),
    }
    for arm, (lbl, col, ls, mk) in style.items():
        series.append((lbl, [arms[arm]['per_trace'][t]['qoe_quality'] for t, _ in TRACES],
                       col, ls, mk))

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    n = len(TRACES)
    ys = np.arange(1, n + 1) / n
    for lbl, vals, col, ls, mk in series:
        xs = np.sort(vals)
        lw = 1.5 if lbl.startswith('DQN') else 0.9
        ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0], ys]),
                where='post', color=col, linestyle=ls, linewidth=lw,
                marker=mk, markersize=3, label=lbl)
    ax.set_xlabel('QoE')
    ax.set_ylabel('P(QoE $\\leq$ x)')
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, ncol=1)
    save(fig, 'Fig6', 'qoe_cdf')


# ================================================================ Fig. 7
def fig7_convergence():
    dirs = sorted(glob.glob(os.path.join(
        ROOT, 'logs', 'train_runs', '20260714_131016_full', 'sweep', 'trial_001_s*')))
    per_seed = []
    best_eps, max_eps = [], []
    for d in dirs:
        h = json.load(open(os.path.join(d, 'abr_dqn_train_history.json')))
        ev = h['eval']
        eps = np.array([e['episode'] for e in ev], dtype=float)
        q = np.array([e['qoe_quality'] for e in ev], dtype=float)
        per_seed.append((eps, q))
        best_eps.append(eps[int(np.argmax(q))])
        max_eps.append(eps[-1])

    grid = np.linspace(0, min(max_eps), 200)
    mat = np.vstack([np.interp(grid, eps, q) for eps, q in per_seed])
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    # best-checkpoint-so-far: running max per seed = QoE of the model that
    # checkpoint selection would ship at that point in training
    best_mat = np.maximum.accumulate(mat, axis=1)
    bmean, bstd = best_mat.mean(axis=0), best_mat.std(axis=0)

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot(grid, mean, color=C_GREY, linewidth=0.8, alpha=0.75,
            label='Raw eval QoE (12-seed mean)')
    ax.fill_between(grid, bmean - bstd, bmean + bstd, color=C_PURPLE, alpha=0.18,
                    linewidth=0, label='$\\pm$1 std (12 seeds)')
    ax.plot(grid, bmean, color=C_PURPLE, linewidth=1.4,
            label='Best checkpoint so far')
    ax.axhline(40.6, color=C_BLUE, linestyle='--', linewidth=0.9,
               label='Best fixed arm (40.6)')
    mb = float(np.mean(best_eps))
    ax.axvline(mb, color=C_GREY, linestyle=':', linewidth=0.9)
    ax.annotate('mean best\ncheckpoint', xy=(mb, 12),
                xytext=(mb + grid[-1] * 0.04, 8), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', linewidth=0.6))
    ax.set_xlabel('Training episode')
    ax.set_ylabel('QoE')
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    save(fig, 'Fig7', 'training_convergence')
    print(f'  mean best-checkpoint episode: {mb:.0f} of {min(max_eps):.0f} '
          f'({100 * mb / min(max_eps):.0f}%)')


if __name__ == '__main__':
    fig3_per_trace_bars()
    fig4_timeseries()
    fig5_segment_curve()
    fig6_cdf()
    fig7_convergence()
    print('done')
