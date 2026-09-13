# -*- coding: utf-8 -*-
"""
ui_components.py
=================
دوال مساعدة لعرض/رسم البيانات داخل واجهة Streamlit (بدون منطق النموذج نفسه).
حالياً تحتوي على دالة رسم خريطة الانتباه الحرارية، ويمكن إضافة أي دوال عرض أخرى هنا.
"""

import matplotlib.pyplot as plt


def plot_attention_heatmap(attentions, tokens, layer_idx: int, head_idx: int, max_tokens: int = 25):
    """
    Self-Attention: لكل توكن، يحسب النموذج مدى "انتباهه" لبقية التوكنز في الجملة
    عند بناء تمثيله الداخلي. الخريطة الحرارية تُظهر قوة هذا الانتباه بين كل زوج توكنز.
    """
    tokens = tokens[:max_tokens]
    attn_matrix = attentions[layer_idx][0, head_idx][:max_tokens, :max_tokens].detach().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn_matrix, cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key Tokens (يُنتبَه إليها)")
    ax.set_ylabel("Query Tokens (تنتبه)")
    ax.set_title(f"Self-Attention — Layer {layer_idx + 1}, Head {head_idx + 1}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
