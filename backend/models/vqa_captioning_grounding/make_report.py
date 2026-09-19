"""
backend/models/vqa_captioning_grounding/make_report.py
"""

import os
import sys
import json
import matplotlib.pyplot as plt

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
RESULTS_JSON = os.path.join(_BACKEND_DIR, "docs", "p2_results.json")
OUTPUT_MD = os.path.join(_BACKEND_DIR, "docs", "p2_evaluation.md")
CHART_PNG = os.path.join(_BACKEND_DIR, "docs", "p2_accuracy_chart.png")

def make_report():
    if not os.path.exists(RESULTS_JSON):
        print(f"Error: {RESULTS_JSON} not found. Run evaluate_p2.py first.")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    if not metrics:
        return

    # Create Chart
    types = []
    maj_accs = []
    base_accs = []
    lora_accs = []

    # Include overall first, then types
    for t in ["overall"] + list(metrics.get("by_type", {}).keys()):
        m = metrics["overall"] if t == "overall" else metrics["by_type"][t]
        types.append(t.capitalize())
        maj_accs.append(m["maj_acc"])
        base_accs.append(m["base_acc"])
        lora_accs.append(m["lora_acc"])

    x = range(len(types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width for i in x], maj_accs, width, label='Majority Class', color='gray')
    ax.bar(x, base_accs, width, label='Base BLIP', color='skyblue')
    ax.bar([i + width for i in x], lora_accs, width, label='RS-LoRA BLIP', color='orange')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('P2 Domain Adaptation Accuracy by Question Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(CHART_PNG)
    plt.close()

    # Generate Markdown Report
    md = [
        "# P2 Evaluation Report",
        "",
        "This report is automatically generated from `p2_results.json`.",
        "",
        "## Overall Metrics",
        "",
        f"- **Total Questions:** {metrics['overall']['n']}",
        f"- **Majority Class Baseline:** {metrics['overall']['maj_acc']:.1f}%",
        f"- **Base BLIP Zero-Shot:** {metrics['overall']['base_acc']:.1f}% (95% CI: [{metrics['overall']['base_ci'][0]:.1f}, {metrics['overall']['base_ci'][1]:.1f}])",
        f"- **RS-LoRA BLIP:** {metrics['overall']['lora_acc']:.1f}% (95% CI: [{metrics['overall']['lora_ci'][0]:.1f}, {metrics['overall']['lora_ci'][1]:.1f}])",
        "",
        "## Metrics by Question Type",
        "",
        "| Question Type | N | Majority Class | Base BLIP | RS-LoRA BLIP |",
        "|---|---|---|---|---|"
    ]

    for t, m in metrics.get("by_type", {}).items():
        row = f"| {t.capitalize()} | {m['n']} | {m['maj_acc']:.1f}% | {m['base_acc']:.1f}% | {m['lora_acc']:.1f}% |"
        md.append(row)

    md.extend([
        "",
        "## Counting Specific Metrics",
        "",
        "For 'count' type questions (where exact match is difficult):",
        "",
        "| Metric | Base BLIP | RS-LoRA BLIP |",
        "|---|---|---|"
    ])

    if "count" in metrics.get("by_type", {}):
        cm = metrics["by_type"]["count"]
        b_mae = f"{cm['base_mae']:.2f}" if cm.get('base_mae') is not None else "N/A"
        l_mae = f"{cm['lora_mae']:.2f}" if cm.get('lora_mae') is not None else "N/A"
        b_w1 = f"{cm['base_within_1_acc']:.1f}%" if cm.get('base_within_1_acc') is not None else "N/A"
        l_w1 = f"{cm['lora_within_1_acc']:.1f}%" if cm.get('lora_within_1_acc') is not None else "N/A"
        
        md.append(f"| Mean Absolute Error (MAE) | {b_mae} | {l_mae} |")
        md.append(f"| Within ±1 Accuracy | {b_w1} | {l_w1} |")
        
    md.extend([
        "",
        "## Accuracy Chart",
        "",
        "![Accuracy Chart](./p2_accuracy_chart.png)"
    ])

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Report generated: {OUTPUT_MD}")
    print(f"Chart generated: {CHART_PNG}")

if __name__ == "__main__":
    make_report()
