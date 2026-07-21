# -*- coding: utf-8 -*-
"""Generate the paper result charts (Fig. 3-7) in IEEE single-column style.

Regenerated for the request-pacing objective (qoe_v4). Baselines are the
Buffer-based and MPC controllers; DQN is the proposed policy.

Outputs PNG (300 dpi) + vector PDF per chart under docs/figures/Fig{3..7}/.
Data sources:
  models/final_test_results_qoe_v4_request_pacing.json  (per-trace + case-level)
  models/dqn_sweep_results_qoe_v4_request_pacing.json   (segment-size screen)
  logs/dqn_fig4_driving_B_2019.12.16_14.23.32/results.csv  (policy timeseries)
  logs/train_runs/20260720_005851_wide/.../trial_011_s*   (convergence, 12 seeds)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import glob
import json
import os
from collections import defaultdict

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

# 6-tier G-PCC ladder (Mbit/s), highest tier = rep 0
TIER_BITRATE = {0: 102.4, 1: 59.6, 2: 35.5, 3: 14.1, 4: 3.4, 5: 0.9}

# proposed + the two ABR baselines
METHODS = [
    ('dqn', 'DQN (proposed)', C_PURPLE, None),
    ('buffer', 'Buffer-based', C_BLUE, '//'),
    ('mpc', 'MPC', C_ORANGE, '..'),
]

# 4 held-out test traces (order = per_trace key order)
TRACES = [
    ('driving_B_2019.12.16_11.49.59.csv', 'Driving A\n(blackout)'),
    ('driving_B_2019.12.16_14.23.32.csv', 'Driving B'),
    ('driving_B_2020.02.14_07.29.00.csv', 'Driving C'),
    ('static_B_2020.02.13_13.57.29.csv', 'Static'),
]

# --- Fig. 4 policy-timeseries run (v4 policy on a driving trace) ---
# alternatives rendered under docs/figures/Fig4/candidates/ (see preview_fig4.py)
FIG4_RUN = os.path.join(ROOT, 'logs', 'dqn_fig4_driving_B_2020.02.13_15.02.01',
                        'results.csv')
FIG4_TRACE = 'driving_B_2020.02.13_15.02.01.csv'

# --- Fig. 7 convergence: winner trial 011, 12 confirm seeds ---
FIG7_SEEDS = os.path.join(
    ROOT, 'logs', 'train_runs', '20260720_005851_wide', 'sweep', 'confirm',
    'trial_011_s*', 'abr_dqn_train_history.json')

results = json.load(open(os.path.join(
    ROOT, 'models', 'final_test_results_qoe_v4_request_pacing.json')))['results']
sweep = json.load(open(os.path.join(
    ROOT, 'models', 'dqn_sweep_results_qoe_v4_request_pacing.json')))


def per_trace(method, metric):
    pt = results[method]['per_trace']
    return np.array([pt[t][metric] for t, _ in TRACES], dtype=float)


def overall(method, metric):
    vals = per_trace(method, metric)
    return vals.mean(), vals.std()


def case_values(method, metric):
    return np.array([c[metric] for c in results[method]['cases']], dtype=float)


def save(fig, sub, name):
    d = os.path.join(FIGS, sub)
    os.makedirs(d, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(d, f'{name}.{ext}'))
    plt.close(fig)
    print(f'saved {sub}/{name}.png|.pdf')


# ================================================================ Fig. 3
def _paired_deltas():
    """Per-scenario QoE gain of DQN (mean over training seeds) minus each
    baseline, matched on (sequence, window, offset, eval seed)."""
    def key(c):
        return (c['sequence'], c['window_id'], c['offset'], c['eval_seed'])
    dq = defaultdict(list)
    for c in results['dqn']['cases']:
        dq[key(c)].append(c['qoe_quality'])
    dqm = {k: np.mean(v) for k, v in dq.items()}
    out = {}
    for base in ('mpc', 'buffer'):
        bl = {key(c): c['qoe_quality'] for c in results[base]['cases']}
        out[base] = np.array([dqm[k] - bl[k] for k in dqm if k in bl])
    return out


def fig3_per_trace_bars():
    """Per-trace QoE: DQN vs Buffer-based vs MPC, plus an Overall group.

    Clean solid-fill grouped bars (no hatching) with white separators and
    value labels; DQN carries a thin dark outline for emphasis.
    """
    labels = [lbl for _, lbl in TRACES] + ['Overall']
    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    for i, (m, lbl, col, _) in enumerate(METHODS):
        vals = per_trace(m, 'qoe_quality')
        heights = np.append(vals, overall(m, 'qoe_quality')[0])
        bars = ax.bar(x + (i - 1) * w, heights, w, label=lbl, color=col,
                      edgecolor='#333333', linewidth=0.5, zorder=3)
        for b in bars:
            v = b.get_height()
            ax.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                        textcoords='offset points', xytext=(0, 1.6),
                        ha='center', fontsize=6, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('QoE')
    ax.set_ylim(0, 80)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False, ncol=3, columnspacing=1.0,
              handlelength=1.1, handletextpad=0.5, bbox_to_anchor=(0.0, 1.13))
    save(fig, 'Fig3', 'per_trace_qoe')


def fig3_paired_winrate():
    """Head-to-head: how often, and by how much, DQN beats each baseline on
    the same matched evaluation scenarios."""
    deltas = _paired_deltas()
    rows = [('mpc', 'vs MPC', C_ORANGE), ('buffer', 'vs Buffer-based', C_BLUE)]
    n = len(next(iter(deltas.values())))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(COL_W, 3.1),
        gridspec_kw={'height_ratios': [1, 2.1], 'hspace': 0.55})

    # --- panel 1: win-rate stacked horizontal bars ---
    for i, (base, lbl, col) in enumerate(rows):
        win = 100.0 * np.mean(deltas[base] > 0)
        ax1.barh(i, win, color=C_GREEN, edgecolor='black', linewidth=0.4)
        ax1.barh(i, 100 - win, left=win, color=C_GREY, alpha=0.30,
                 edgecolor='black', linewidth=0.4)
        ax1.text(win - 1.5, i, f'{win:.0f}% win', ha='right', va='center',
                 fontsize=7, color='white', fontweight='bold')
        ax1.text(win + 1.5, i, f'{100 - win:.0f}% loss', ha='left',
                 va='center', fontsize=6.3, color=C_GREY)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels([lbl for _, lbl, _ in rows])
    ax1.set_xlim(0, 100)
    ax1.set_xlabel(f'DQN win rate  (% of {n} matched scenarios)', fontsize=7)
    ax1.invert_yaxis()
    ax1.tick_params(length=0)
    for s in ('top', 'right', 'left'):
        ax1.spines[s].set_visible(False)

    # --- panel 2: per-case QoE-gain distribution ---
    lo = min(d.min() for d in deltas.values())
    bins = np.arange(np.floor(lo / 5) * 5, 22.5, 2.5)
    ax2.axvspan(0, bins[-1], color=C_GREEN, alpha=0.06, linewidth=0)
    peak = 0
    for base, lbl, col in rows:
        d = deltas[base]
        counts, _, _ = ax2.hist(d, bins=bins, histtype='step', color=col,
                                linewidth=1.3,
                                label=f'{lbl} (median +{np.median(d):.1f})')
        peak = max(peak, counts.max())
        ax2.axvline(np.median(d), color=col, linestyle=':', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_ylim(0, peak * 1.32)
    ax2.annotate('DQN better $\\rightarrow$', xy=(bins[-1] * 0.62, peak * 1.2),
                 fontsize=6.8, ha='center', color=C_GREY)
    ax2.annotate('blackout\nlosses', xy=(lo * 0.78, peak * 0.28), fontsize=6.2,
                 ha='center', color=C_VERM)
    ax2.set_xlabel('Per-case QoE gain,  DQN $-$ baseline')
    ax2.set_ylabel('Scenarios')
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', frameon=False)
    save(fig, 'Fig3', 'qoe_paired_winrate')


# ================================================================ Fig. 4
def plot_policy_timeseries(run_csv, trace):
    """Two-panel session view (Fig3-consistent styling): top = per-segment
    DQN tier bitrate as bars under the link-capacity line; bottom = buffer +
    stalls. Returns the Figure so callers can save it anywhere."""
    rows = list(csv.DictReader(open(run_csv)))
    # per-segment start time (cumulative at previous segment end) + tier bitrate
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
    t_end = prev_end
    seg_t = np.array(seg_t, dtype=float)
    seg_rate = np.array(seg_rate, dtype=float)
    seg_w = np.diff(np.append(seg_t, t_end))   # each bar spans its download window

    # capacity from the trace CSV (1 sample/s), same wall-clock axis
    tr = list(csv.DictReader(open(glob.glob(os.path.join(
        ROOT, 'bandwidth_5g', '**', trace), recursive=True)[0])))
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
                                   gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.14})
    # top: selected tier as bars (Fig3 aesthetic) + capacity envelope line
    ax1.bar(seg_t, seg_rate, width=seg_w, align='edge', color=C_PURPLE,
            edgecolor='#333333', linewidth=0.3, zorder=2, label='DQN tier bitrate')
    ax1.step(np.append(cap_t, t_end), np.append(cap, cap[-1]), where='post',
             color=C_BLUE, linewidth=1.1, zorder=3, label='Link capacity')
    ax1.set_ylabel('Rate (Mbit/s)')
    ax1.set_ylim(bottom=0)
    ax1.yaxis.grid(True, alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper left', frameon=False)

    ax2.plot(buf_t, buf, color=C_GREEN, linewidth=1.1, label='Buffer level')
    for a, b in stall_spans:
        ax2.axvspan(a, b, color=C_VERM, alpha=0.3, linewidth=0)
    if stall_spans:
        ax2.axvspan(0, 0, color=C_VERM, alpha=0.3, label='Stall')
    ax2.set_ylabel('Buffer (s)')
    ax2.set_xlabel('Download wall-clock time (s)')
    ax2.set_ylim(0, 5.3)
    ax2.yaxis.grid(True, alpha=0.35)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper left', frameon=False)
    ax2.set_xlim(0, t_end)
    return fig


def fig4_timeseries():
    fig = plot_policy_timeseries(FIG4_RUN, FIG4_TRACE)
    save(fig, 'Fig4', 'policy_timeseries')


# ================================================================ Fig. 5
def fig5_segment_curve():
    """Segment-size sensitivity from the DQN screening sweep (best trial per S)."""
    by_s = {}
    for t in sweep['screening_ranking']:
        S = int(t['config']['segment-frames'])
        met = t['metrics']
        by_s.setdefault(S, []).append((met['qoe_quality'], met['stall_s']))
    S = sorted(by_s)
    qoe = [max(by_s[s])[0] for s in S]
    # stall of the best-QoE trial at each S
    stall = [max(by_s[s], key=lambda p: p[0])[1] for s in S]
    best_s = S[int(np.argmax(qoe))]

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.axvspan(best_s - 1.2, best_s + 1.2, color=C_GREEN, alpha=0.12, linewidth=0)
    l1, = ax.plot(S, qoe, marker='o', markersize=4, color=C_PURPLE, label='QoE (left)')
    ax.set_xlabel('Segment size S (frames per request)')
    ax.set_ylabel('QoE')
    ax.set_xticks(S)
    ax.set_ylim(min(qoe) - 0.6, max(qoe) + 0.4)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    l2, = ax2.plot(S, stall, marker='s', markersize=3.5, linestyle=(0, (4, 2)),
                   color=C_VERM, label='Stall (right)')
    ax2.set_ylabel('Stall (s)')
    ax2.set_ylim(bottom=0)

    ax.annotate('sweet spot', xy=(best_s, max(qoe)),
                xytext=(best_s + 1.2, max(qoe) - 2.3),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', linewidth=0.6))
    ax.legend(handles=[l1, l2], loc='lower right', frameon=False,
              handlelength=2.2)
    save(fig, 'Fig5', 'segment_size_curve')


# ================================================================ Fig. 6
def fig6_cdf():
    """Empirical CDF of per-case QoE for DQN vs the two baselines."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for m, lbl, col, _ in METHODS:
        xs = np.sort(case_values(m, 'qoe_quality'))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        lw = 1.6 if m == 'dqn' else 1.0
        ax.step(np.concatenate([[xs[0]], xs]), np.concatenate([[0], ys]),
                where='post', color=col, linewidth=lw,
                label=f'{lbl} (n={len(xs)})')
        med = np.median(xs)
        ax.plot([med], [0.5], marker='o', markersize=3.5, color=col)
    ax.set_xlabel('QoE (per evaluation case)')
    ax.set_ylabel('P(QoE $\\leq$ x)')
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, color=C_GREY, linewidth=0.5, linestyle=':')
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'Fig6', 'qoe_cdf')


# ================================================================ Fig. 7
def fig7_convergence():
    dirs = sorted(glob.glob(FIG7_SEEDS))
    per_seed = []
    best_eps, max_eps = [], []
    for path in dirs:
        h = json.load(open(path))
        ev = h['validation']
        eps = np.array([e['episode'] for e in ev], dtype=float)
        q = np.array([e['qoe_quality'] for e in ev], dtype=float)
        per_seed.append((eps, q))
        best_eps.append(eps[int(np.argmax(q))])
        max_eps.append(eps[-1])

    grid = np.linspace(0, min(max_eps), 200)
    mat = np.vstack([np.interp(grid, eps, q) for eps, q in per_seed])
    mean = mat.mean(axis=0)
    # best-checkpoint-so-far: running max per seed = QoE of the model that
    # checkpoint selection would ship at that point in training
    best_mat = np.maximum.accumulate(mat, axis=1)
    bmean, bstd = best_mat.mean(axis=0), best_mat.std(axis=0)

    base_q = max(overall('buffer', 'qoe_quality')[0],
                 overall('mpc', 'qoe_quality')[0])
    base_lbl = 'MPC' if overall('mpc', 'qoe_quality')[0] >= overall(
        'buffer', 'qoe_quality')[0] else 'Buffer-based'

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.plot(grid, mean, color=C_GREY, linewidth=0.8, alpha=0.75,
            label=f'Raw eval ({len(dirs)} seeds)')
    ax.fill_between(grid, bmean - bstd, bmean + bstd, color=C_PURPLE, alpha=0.18,
                    linewidth=0, label='$\\pm$1 std')
    ax.plot(grid, bmean, color=C_PURPLE, linewidth=1.4,
            label='Best checkpoint')
    ax.axhline(base_q, color=C_BLUE, linestyle='--', linewidth=0.9,
               label=f'Best baseline ({base_lbl})')
    mb = float(np.mean(best_eps))
    ax.axvline(mb, color=C_GREY, linestyle=':', linewidth=0.9)
    ax.annotate('mean best\ncheckpoint', xy=(mb, bmean.min()),
                xytext=(mb + grid[-1] * 0.04, bmean.min() - 3), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', linewidth=0.6))
    ax.set_xlabel('Training episode')
    ax.set_ylabel('QoE')
    ax.set_ylim(top=bmean.max() + 0.6)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    # legend above the axes (like Fig3) so it never covers the curves/band
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), frameon=False,
              fontsize=6.5, ncol=2, columnspacing=1.3, handlelength=1.7,
              handletextpad=0.5)
    save(fig, 'Fig7', 'training_convergence')
    print(f'  mean best-checkpoint episode: {mb:.0f} of {min(max_eps):.0f} '
          f'({100 * mb / min(max_eps):.0f}%)')


if __name__ == '__main__':
    fig3_per_trace_bars()
    fig3_paired_winrate()
    fig4_timeseries()
    fig5_segment_curve()
    fig6_cdf()
    fig7_convergence()
    print('done')
