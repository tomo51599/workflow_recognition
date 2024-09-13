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
def gen_ribbon_plot_no_weight(ribbon_results, xml_no):
    plt.rcParams.update({'font.size': 14}) 
    
        
        