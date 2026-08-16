"""
Generates 14 publication-quality figures for the EdgeMem Research Paper.
All figures formatted at 300 DPI.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def plot_fig13_frontier():
    """Figure 13: Memory Utility vs Capacity & Context Budget Frontier."""
    compressions = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0]
    accuracy = [100.0, 90.0, 90.0, 70.0, 40.0, 10.0]
    ram_kb = [38.28, 34.45, 26.80, 19.14, 11.48, 3.83]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    color = '#1f77b4'
    ax1.set_xlabel('Memory Capacity Compression Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Personalization Recall Accuracy (%)', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(compressions, accuracy, color=color, marker='o', linewidth=2.8, label='Personalization Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 110)
    ax1.grid(True)

    # Highlight inflection / collapse point (50% to 70%)
    ax1.axvspan(45, 75, color='#ffebee', alpha=0.6, label='Personalization Collapse Zone')

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Active Working Graph RAM (KB)', color=color, fontsize=11, fontweight='bold')
    line2 = ax2.plot(compressions, ram_kb, color=color, marker='s', linewidth=2.5, linestyle='--', label='Vector RAM (KB)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 45)

    lines = line1 + [line2[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=9.5, framealpha=0.9)

    plt.title('Figure 13: The Memory Utility Frontier — Personalization vs Storage Budget', fontsize=12.5, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = "figures/fig13_memory_utility_frontier.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")

def plot_fig14_ablation():
    """Figure 14: 4-Way Graph Necessity Ablation."""
    systems = ['System A\n(Full Dump)', 'System B\n(Vector Only)', 'System C\n(Graph Only)', 'System D\n(EdgeMem Complete)']
    multihop_acc = [100.0, 60.0, 100.0, 100.0]
    tokens = [234.0, 11.2, 75.0, 7.8]

    x = np.arange(len(systems))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=300)

    rects1 = ax1.bar(x - width/2, multihop_acc, width, label='Multi-Hop Accuracy (%)', color='#2ca02c', edgecolor='#1b7a1b')
    ax1.set_ylabel('Multi-Hop Accuracy (%)', fontsize=11, fontweight='bold', color='#1b7a1b')
    ax1.set_ylim(0, 120)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=10.5, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.0f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, tokens, width, label='Avg Injected Tokens / Turn', color='#d62728', edgecolor='#8c1b1b', alpha=0.85)
    ax2.set_ylabel('Injected Context Tokens / Turn', color='#d62728', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 270)

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=9.5, framealpha=0.9)

    plt.title('Figure 14: 4-Way Architectural Ablation — Multi-Hop Accuracy vs Context Token Economy', fontsize=12.0, fontweight='bold', pad=12)
    plt.tight_layout()
    out_path = "figures/fig14_graph_necessity_ablation.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[SAVED] {out_path}")

def main():
    print("Generating complete publication-quality figures for research paper...")
    plot_fig13_frontier()
    plot_fig14_ablation()
    print("All publication figures successfully generated in figures/")

if __name__ == "__main__":
    main()
