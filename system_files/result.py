import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

#結果をカウントする関数
def count_prediction(predicted_phase, current_phase, results):
    if current_phase is not None:
        if current_phase in results and predicted_phase in results[current_phase]:
            results[current_phase][predicted_phase] += 1
    
    return results

#混合行列を生成する関数
def gen_conf_matrix(results, xml_no, mode):
    
    size = len(results)
    conf_matrix = np.zeros((size, size))
    
    for actual, predicted_counts in results.items():
        for predicted, count in predicted_counts.items():
            conf_matrix[actual -1, predicted -1] = count
    
    conf_matrix = conf_matrix.astype(int)        
            
    plt.figure(figsize = (10, 8))
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap= "viridis", xticklabels=[f'phase{i}' for i in range(1, size+1)], yticklabels=[f'phase{i}' for i in range(1, size+1)])
    plt.ylabel('Actual Phase')
    plt.title(f'Confusion Matrix of Action Recognition for Test_{xml_no}')
    plt.savefig(f'/home/master_thesis/results/confusion_matrix_{xml_no}_{mode}.png')    
    
    plt.figure()        
    
#リボン図を生成する関数(重みなし)
def gen_ribbon_plot_no_weight(prediction_dict, xml_no):
    plt.rcParams.update({'font.size': 14}) 
    
    colors = ['grey', 'orange', 'yellow', 'blue', 'black', 'green', 'purple']
    
    frames = sorted(prediction_dict.keys())
    
    predicted_labels = [int(prediction_dict[frame]['label'][0]) if isinstance(prediction_dict[frame]['label'], np.ndarray) else int(prediction_dict[frame]['label']) for frame in frames]
    second_labels = [int(prediction_dict[frame]['second_label']) for frame in frames]
    actual_labels = [int(prediction_dict[frame]['actual_label']) if prediction_dict[frame]['actual_label'] is not None else 0 for frame in frames]
    
    fig, axs = plt.subplots(3, 1, figsize= (15, 3), sharex = True)
    
    for i, frame in enumerate(frames):
        predicted_y = 0.5
        axs[0].add_patch(Rectangle((frame, predicted_y), 48, 1, color = colors[predicted_labels[i]]))
        
    for i, frame in enumerate(frames):
        second_y = 0.5
        axs[1].add_patch(Rectangle((frame, second_y), 48, 1, color = colors[second_labels[i]]))
        
    for i, frame in enumerate(frames):
        actual_y = 0.5
        axs[2].add_patch(Rectangle((frame, actual_y), 48, 1, color = colors[actual_labels[i]])) 
        
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Predicted', fontsize=12, labelpad=20)

    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Second', fontsize=12, labelpad=20)
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Actual', fontsize=12, labelpad=20)
    axs[2].set_xlabel('Frame Number (Time)')  
    
    axs[0].set_xlim(min(frames), max(frames))         
        
    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])    
    
    handles = [
    mpatches.Patch(color=colors[0], label='Not a Phase'),
    mpatches.Patch(color=colors[1], label='Phase 1'),
    mpatches.Patch(color=colors[2], label='Phase 2'),
    mpatches.Patch(color=colors[3], label='Phase 3'),
    mpatches.Patch(color=colors[4], label='Phase 4'),
    mpatches.Patch(color=colors[5], label='Phase 5'),
    mpatches.Patch(color=colors[6], label='Phase 6')
    ]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.suptitle(f'Ribbon Plot_NO_weight No.{xml_no}')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, hspace=0.05)
    plt.savefig(f'master_thesis/results/ribbon_plot_{xml_no}_no_weight.png', bbox_inches='tight', pad_inches=0)
    
    