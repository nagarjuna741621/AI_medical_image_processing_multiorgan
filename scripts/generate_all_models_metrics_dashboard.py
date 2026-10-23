#!/usr/bin/env python3
"""
Master Metrics & Comparative Benchmark Dashboard Generator
===========================================================
Generates publication-quality head-to-head comparison charts and performance
registries across DenseNet-121, CoAtNet, MobileViT, and EfficientNetB0.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Ensure repository root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.config import OUTPUTS_DIR

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

JSON_PATH = os.path.join(OUTPUTS_DIR, 'downloaded_models_evaluation_report.json')

if not os.path.exists(JSON_PATH):
    print(f"Error: Evaluation report not found at {JSON_PATH}")
    sys.exit(1)

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

chest_models = data['chest_models']
brain_models = data['brain_models']

# Explicitly ensure exact model names and distinguish Original Model
for m in chest_models + brain_models:
    base_file = os.path.basename(m['model_path'])
    if base_file in ['chest_model.keras', 'brain_model.keras']:
        m['model_name'] = 'Original Model (CNN - EfficientNetB0)'
    elif 'densenet121' in base_file:
        m['model_name'] = 'DenseNet-121' if 'brain' in base_file else 'DenseNet-121 (CheXNet)'
    elif 'coatnet' in base_file:
        m['model_name'] = 'CoAtNet (Hybrid)'
    elif 'mobilevit' in base_file:
        m['model_name'] = 'MobileViT (Edge Hybrid)'

# Distinct, professional color palette
MODEL_COLORS = {
    'Original Model (CNN - EfficientNetB0)': '#e63946',  # Bright Crimson
    'DenseNet-121 (CheXNet)': '#1d3557',               # Deep Navy
    'DenseNet-121': '#1d3557',                         # Deep Navy
    'MobileViT (Edge Hybrid)': '#2a9d8f',              # Emerald Teal
    'CoAtNet (Hybrid)': '#f4a261'                      # Amber Gold
}

MODEL_LINESTYLES = {
    'Original Model (CNN - EfficientNetB0)': '--',
    'DenseNet-121 (CheXNet)': '-',
    'DenseNet-121': '-',
    'MobileViT (Edge Hybrid)': '-.',
    'CoAtNet (Hybrid)': ':'
}

# Create Figure
fig = plt.figure(figsize=(24, 17), facecolor='#ffffff')
gs = GridSpec(3, 2, height_ratios=[1.2, 1.25, 0.95], hspace=0.35, wspace=0.25)

metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Macro F1', 'ROC-AUC']
n_metrics = len(metrics)
n_models = len(chest_models)
bar_width = 0.18
x = np.arange(n_metrics)

# 1. Top-Left: Chest X-Ray Multi-Metric Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
for i, m in enumerate(chest_models):
    name = m['model_name']
    vals = [m[k] for k in metrics]
    color = MODEL_COLORS.get(name, '#6c757d')
    offset = (i - (n_models - 1) / 2) * bar_width
    is_orig = 'Original' in name
    edge_col = '#991b1b' if is_orig else 'white'
    lw = 2.2 if is_orig else 1.0
    rects = ax1.bar(x + offset, vals, width=bar_width, label=name, color=color, edgecolor=edge_col, linewidth=lw, alpha=0.92)
    for rect, val in zip(rects, vals):
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., h + 0.015, f"{val:.1%}" if val < 0.99 else f"{val:.3f}",
                 ha='center', va='bottom', fontsize=8.0, fontweight='bold', rotation=0, color='#0f172a')

ax1.set_xticks(x)
ax1.set_xticklabels(metric_labels, fontweight='bold', fontsize=11)
ax1.set_ylim(0.0, 1.08)
ax1.set_title("A. Chest X-Ray Multi-Metric Benchmark: Original Model vs. Downloaded Models", fontweight='bold', fontsize=13, pad=12, color='#0f172a')
ax1.set_ylabel("Metric Score", fontweight='bold', fontsize=11)
ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.5)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# 2. Top-Right: Brain MRI Multi-Metric Bar Chart
ax2 = fig.add_subplot(gs[0, 1])
for i, m in enumerate(brain_models):
    name = m['model_name']
    vals = [m[k] for k in metrics]
    color = MODEL_COLORS.get(name, '#6c757d')
    offset = (i - (n_models - 1) / 2) * bar_width
    is_orig = 'Original' in name
    edge_col = '#991b1b' if is_orig else 'white'
    lw = 2.2 if is_orig else 1.0
    rects = ax2.bar(x + offset, vals, width=bar_width, label=name, color=color, edgecolor=edge_col, linewidth=lw, alpha=0.92)
    for rect, val in zip(rects, vals):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2., h + 0.015, f"{val:.1%}" if val < 0.99 else f"{val:.3f}",
                 ha='center', va='bottom', fontsize=8.0, fontweight='bold', rotation=0, color='#0f172a')

ax2.set_xticks(x)
ax2.set_xticklabels(metric_labels, fontweight='bold', fontsize=11)
ax2.set_ylim(0.0, 1.08)
ax2.set_title("B. Brain MRI Multi-Metric Benchmark: Original Model vs. Downloaded Models", fontweight='bold', fontsize=13, pad=12, color='#0f172a')
ax2.set_ylabel("Metric Score", fontweight='bold', fontsize=11)
ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cbd5e1', fontsize=9.5)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# 3. Middle-Left: Multi-Model Radar Diagnostic Profile (Chest X-Ray)
ax3 = fig.add_subplot(gs[1, 0], polar=True)
radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
radar_labels = ['Accuracy', 'Precision', 'Sensitivity\n(Recall)', 'F1-Score', 'ROC-AUC']
num_vars = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

for m in chest_models:
    name = m['model_name']
    values = [m[k] for k in radar_metrics]
    values += values[:1]
    color = MODEL_COLORS.get(name, '#6c757d')
    ls = MODEL_LINESTYLES.get(name, '-')
    is_orig = 'Original' in name
    lw = 3.2 if is_orig else 2.2
    alpha = 0.08 if is_orig else 0.12
    ax3.plot(angles, values, color=color, linewidth=lw, linestyle=ls, label=name, marker='o' if is_orig else 's', markersize=5)
    ax3.fill(angles, values, color=color, alpha=alpha)

ax3.set_theta_offset(np.pi / 2)
ax3.set_theta_direction(-1)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, fontweight='bold', fontsize=10, color='#0f172a')
ax3.set_ylim(0.80, 1.00)
ax3.set_title("C. Multi-Model Radar Diagnostic Profile (Original Baseline vs. Downloaded)", fontweight='bold', fontsize=13, pad=20, color='#0f172a')
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.08), frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)

# 4. Middle-Right: Clinical Efficiency Frontier (Accuracy vs. Latency vs. Size)
ax4 = fig.add_subplot(gs[1, 1])

for m in chest_models:
    name = m['model_name']
    acc = m['accuracy'] * 100
    lat = m['latency_ms']
    size = m['file_size_mb']
    color = MODEL_COLORS.get(name, '#6c757d')
    is_orig = 'Original' in name
    edge = '#991b1b' if is_orig else '#334155'
    lw = 2.5 if is_orig else 1.2
    ax4.scatter(lat, acc, s=size * 35 + 80, color=color, alpha=0.88 if is_orig else 0.75, edgecolors=edge, linewidth=lw, zorder=4)

annotations_config = {
    'MobileViT (Edge Hybrid)': {
        'xytext': (19.9, 92.4),
        'ha': 'center', 'va': 'bottom',
        'desc': 'MobileViT (Edge Hybrid)\nAcc: 90.6% | Lat: 19.9ms | 10.8MB\n[Lowest Latency / Fastest]'
    },
    'CoAtNet (Hybrid)': {
        'xytext': (25.2, 88.4),
        'ha': 'center', 'va': 'top',
        'desc': 'CoAtNet (Hybrid)\nAcc: 89.7% | Lat: 25.2ms | 11.2MB'
    },
    'Original Model (CNN - EfficientNetB0)': {
        'xytext': (41.0, 91.8),
        'ha': 'center', 'va': 'bottom',
        'desc': '★ ORIGINAL MODEL (CNN)\nAcc: 90.0% | Lat: 39.5ms | 18.2MB\n[Project Baseline Reference]'
    },
    'DenseNet-121 (CheXNet)': {
        'xytext': (73.0, 91.2),
        'ha': 'center', 'va': 'top',
        'desc': 'DenseNet-121 (CheXNet)\nAcc: 93.8% | Lat: 73.0ms | 36.3MB\n[Highest Diagnostic Accuracy]'
    }
}

for m in chest_models:
    name = m['model_name']
    acc = m['accuracy'] * 100
    lat = m['latency_ms']
    is_orig = 'Original' in name
    cfg = annotations_config.get(name, {'xytext': (lat, acc + 0.5), 'ha': 'center', 'va': 'bottom', 'desc': name})
    
    ax4.annotate(
        cfg['desc'],
        xy=(lat, acc),
        xytext=cfg['xytext'],
        ha=cfg['ha'],
        va=cfg['va'],
        fontweight='bold',
        fontsize=9.0,
        color='#b91c1c' if is_orig else '#0f172a',
        bbox=dict(boxstyle="round,pad=0.4", fc="#fee2e2" if is_orig else "#f8fafc", ec="#ef4444" if is_orig else "#cbd5e1", lw=1.2, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color='#b91c1c' if is_orig else '#475569', lw=1.5 if is_orig else 1.2),
        zorder=5
    )

ax4.set_xlabel("CPU Inference Latency per Scan (ms) [Lower = Faster Real-Time Triage]", fontweight='bold', fontsize=11)
ax4.set_ylabel("Diagnostic Accuracy (%) [Higher = Superior Clinical Sensitivity]", fontweight='bold', fontsize=11)
ax4.set_title("D. Efficiency Frontier: Original Model Baseline vs. Downloaded Candidates", fontweight='bold', fontsize=13, pad=12, color='#0f172a')
ax4.set_xlim(10, 85)
ax4.set_ylim(87.2, 95.2)
ax4.axvline(x=20, color='#10b981', linestyle=':', linewidth=1.5, alpha=0.85, label='Real-Time Edge Target (<=20ms)')
ax4.axhspan(92.0, 95.0, color='#10b981', alpha=0.06)
ax4.text(12, 94.4, "High Accuracy Clinical Zone (>=92%)", color='#047857', fontweight='bold', fontsize=9.5)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#cbd5e1', fontsize=9.5)

# 5. Bottom: Master Performance Registry Table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

headers = [
    "Organ Task", "Architecture / Role", "Accuracy", "Precision",
    "Recall (Sens.)", "Macro F1", "ROC-AUC", "Latency (CPU)", "Disk Size", "Parameters"
]

col_widths = [0.10, 0.24, 0.08, 0.08, 0.09, 0.08, 0.08, 0.09, 0.08, 0.08]

table_data = []
row_is_original = []

for m in chest_models:
    is_orig = 'Original' in m['model_name']
    row_is_original.append(is_orig)
    table_data.append([
        "Chest X-Ray",
        ("★ " if is_orig else "") + m['model_name'],
        f"{m['accuracy']:.2%}",
        f"{m['precision']:.2%}",
        f"{m['recall']:.2%}",
        f"{m['f1_score']:.4f}",
        f"{m['roc_auc']:.4f}",
        f"{m['latency_ms']:.1f} ms",
        f"{m['file_size_mb']:.1f} MB",
        f"{m['parameters']:,}"
    ])

for m in brain_models:
    is_orig = 'Original' in m['model_name']
    row_is_original.append(is_orig)
    table_data.append([
        "Brain MRI",
        ("★ " if is_orig else "") + m['model_name'],
        f"{m['accuracy']:.2%}",
        f"{m['precision']:.2%}",
        f"{m['recall']:.2%}",
        f"{m['f1_score']:.4f}",
        f"{m['roc_auc']:.4f}",
        f"{m['latency_ms']:.1f} ms",
        f"{m['file_size_mb']:.1f} MB",
        f"{m['parameters']:,}"
    ])

tab = ax5.table(cellText=table_data, colLabels=headers, colWidths=col_widths, loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(9.5)
tab.scale(1.0, 1.45)

for (row, col), cell in tab.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1e293b')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        orig_row = row_is_original[row - 1]
        if orig_row:
            cell.set_facecolor('#fee2e2')
            cell.set_text_props(color='#991b1b', fontweight='bold')
        elif row in [1, 5]:
            cell.set_facecolor('#f0fdf4')
            cell.set_text_props(color='#166534', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f8fafc')
        else:
            cell.set_facecolor('#ffffff')

ax5.set_title("E. Numerical Diagnostic Performance Registry (Original Baseline Highlighted in Red)", fontweight='bold', fontsize=13, pad=10, color='#0f172a')

# Super Title
plt.suptitle("Multi-Organ Medical AI Diagnostic Benchmark: Original Model (CNN) vs. Downloaded Candidates\n"
             "Head-to-Head Evaluation across 4 Architectures: Original Model (CNN - EfficientNetB0), DenseNet-121, CoAtNet, & MobileViT",
             fontsize=17, fontweight='bold', y=0.99, color='#0f172a')

output_path = os.path.join(OUTPUTS_DIR, "all_models_metrics_master_dashboard.png")
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Master metrics comparison dashboard saved successfully to:\n -> {output_path}")
