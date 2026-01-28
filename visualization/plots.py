import re
import numpy as np
from matplotlib import pyplot as pl

class colors:
    red_rgb = (0.8, 0.1, 0.1)
    blue_rgb = (0.1, 0.1, 0.8)

def bar(contributions, feature_names, max_display=10, show=True, title=None, fontsize=13, save_path=None):
    values = contributions
    
    xlabel = "Concept contributions"
    
    if feature_names is None:
        feature_names = [f"Concept {i}" for i in range(len(values))]
    
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values))
    max_display = min(max_display, num_features)
    
    orig_inds = [i for i in range(len(values))]
    
    feature_order = np.argsort(np.abs(values))[::-1]
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos, inds in enumerate(orig_inds):
        feature_names_new.append(feature_names[inds])
    feature_names = feature_names_new
    
    yticklabels = []
    for i in feature_inds:
        yticklabels.append(feature_names[i])
    
    if num_features < len(values):
        num_cut = np.sum([1 for i in range(num_features-1, len(values))])
        values[feature_order[num_features-1]] = np.sum([values[feature_order[i]] for i in range(num_features-1, len(values))], 0)
    
    if num_features < len(values):
        yticklabels[-1] = f"Sum of {num_cut} other features"
    
    row_height = 0.55
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)
    
    negative_values_present = np.sum(values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)
    
    total_width = 0.7
    bar_width = total_width
    
    pl.barh(
        y_pos, values[feature_inds],
        bar_width, align='center',
        color=[colors.blue_rgb if values[feature_inds[j]] <= 0 else colors.red_rgb for j in range(len(y_pos))],
        edgecolor=(1,1,1,0.8)
    )
    
    pl.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)
    
    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    
    for j in range(len(y_pos)):
        ind = feature_order[j]
        if values[ind] < 0:
            pl.text(
                values[ind] - (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=fontsize
            )
        else:
            pl.text(
                values[ind] + (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=fontsize
            )
    
    for i in range(num_features):
        pl.axhline(i+1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)
    
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=fontsize)
    
    xmin, xmax = pl.gca().get_xlim()
    
    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.1)
    
    pl.xlabel(xlabel, fontsize=fontsize)
    if title:
        pl.title(title, fontsize=fontsize)
    
    if save_path:
        pl.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved plot to {save_path}")
    
    if show:
        pl.show()
    else:
        pl.close()


def format_value(s, format_str):
    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s


def plot_weight_heatmap(weights, class_names, concept_names, top_k_concepts=20, save_path=None, show=True):
    num_classes, num_concepts = weights.shape
    
    concept_importance = np.sum(np.abs(weights), axis=0)
    top_concept_indices = np.argsort(concept_importance)[::-1][:top_k_concepts]
    
    weights_subset = weights[:, top_concept_indices]
    concept_names_subset = [concept_names[i] for i in top_concept_indices]
    
    fig, ax = pl.subplots(figsize=(max(12, top_k_concepts * 0.5), max(8, num_classes * 0.4)))
    
    im = ax.imshow(weights_subset, aspect='auto', cmap='RdBu_r', vmin=-np.abs(weights_subset).max(), vmax=np.abs(weights_subset).max())
    
    ax.set_xticks(np.arange(len(concept_names_subset)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(concept_names_subset, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, fontsize=10)
    
    ax.set_xlabel('Concepts', fontsize=12)
    ax.set_ylabel('Classes', fontsize=12)
    ax.set_title(f'Final Layer Weights (Top {top_k_concepts} Concepts)', fontsize=14)
    
    pl.colorbar(im, ax=ax, label='Weight Value')
    pl.tight_layout()
    
    if save_path:
        pl.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved heatmap to {save_path}")
    
    if show:
        pl.show()
    else:
        pl.close()


def plot_top_concepts_per_class(weights, class_names, concept_names, top_k=5, save_path=None, show=True):
    num_classes = len(class_names)
    fig, axes = pl.subplots(num_classes, 1, figsize=(10, num_classes * 1.5))
    
    if num_classes == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        class_weights = weights[i, :]
        top_indices = np.argsort(np.abs(class_weights))[::-1][:top_k]
        top_weights = class_weights[top_indices]
        top_concepts = [concept_names[j] for j in top_indices]
        
        colors_list = [colors.red_rgb if w > 0 else colors.blue_rgb for w in top_weights]
        
        axes[i].barh(range(len(top_concepts)), top_weights, color=colors_list)
        axes[i].set_yticks(range(len(top_concepts)))
        axes[i].set_yticklabels(top_concepts, fontsize=9)
        axes[i].set_xlabel('Weight Value', fontsize=10)
        axes[i].set_title(f'{class_name} - Top {top_k} Concepts', fontsize=11)
        axes[i].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[i].grid(axis='x', alpha=0.3)
    
    pl.tight_layout()
    
    if save_path:
        pl.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved top concepts plot to {save_path}")
    
    if show:
        pl.show()
    else:
        pl.close()
