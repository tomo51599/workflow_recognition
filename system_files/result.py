import os, cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go

from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#結果をカウントする関数
def count_prediction(predicted_phase, current_phase, results):
    if current_phase is not None:
        if current_phase in results and predicted_phase in results[current_phase]:
            results[current_phase][predicted_phase] += 1
    
    return results

#混合行列を生成する関数
def gen_conf_matrix(results, xml_no, mode, frame_phase_changed):
    
    plt.close()
    size = len(results)
    conf_matrix = np.zeros((size, size))
    
    for actual, predicted_counts in results.items():
        for predicted, count in predicted_counts.items():
            conf_matrix[actual -1, predicted -1] = count
    
    conf_matrix = conf_matrix.astype(int)
    cal_evaluation(conf_matrix, xml_no, mode, frame_phase_changed)        
            
    plt.figure(figsize = (10, 8))
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap= "viridis", xticklabels=[f'phase{i}' for i in range(1, size+1)], yticklabels=[f'phase{i}' for i in range(1, size+1)])
    plt.ylabel('Actual Phase')
    plt.xlabel('Predicted Phase')
    plt.title(f'Confusion Matrix of Action Recognition for Test_{xml_no}_{mode}')
    plt.savefig(f'/home/master_thesis/results/confusion_matrix_{xml_no}_{mode}.png')    
    
    plt.figure()     
    
#評価値を計算する関数    
def cal_evaluation(conf_matrix, xml_no, mode, frame_phase_changed):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis = 0) - TP
    FN = np.sum(conf_matrix, axis = 1) - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score) 
    
    log_path = f"master_thesis/results/evaluation{xml_no}_{mode}.log"
    os.makedirs(os.path.dirname(f"master_thesis/results/evaluation_log{xml_no}.txt"), exist_ok=True)
    
    with open(log_path, "w") as log_file:
        log_file.write(f"precision: {precision}\n")
        log_file.write(f"recall: {recall}\n")
        log_file.write(f"f1_score: {f1_score}\n")
        log_file.write(f"{str(frame_phase_changed)}\n")
        log_file.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        
    print(f"precision{precision},recall{recall},f1-score{f1_score}")
    
#混合行列を生成する関数(重みなし)
def gen_conf_matrix_no_weight(results, xml_no):
    
    plt.close()
    size = len(results)
    conf_matrix = np.zeros((size, size))
    
    for actual, predicted_counts in results.items():
        for predicted, count in predicted_counts.items():
            conf_matrix[actual -1, predicted -1] = count
    
    conf_matrix = conf_matrix.astype(int)
    cal_evaluation_no_weight(conf_matrix, xml_no)        
            
    plt.figure(figsize = (10, 8))
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap= "viridis", xticklabels=[f'phase{i}' for i in range(1, size+1)], yticklabels=[f'phase{i}' for i in range(1, size+1)])
    plt.ylabel('Actual Phase')
    plt.xlabel('Predicted Phase')
    plt.title(f'Confusion Matrix of Action Recognition for Test_{xml_no}_no_weight')
    plt.savefig(f'/home/master_thesis/results/confusion_matrix_{xml_no}_no_weight.png')    
    
    plt.figure()     
    
#評価値を計算する関数(重みなし)    
def cal_evaluation_no_weight(conf_matrix, xml_no):
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis = 0) - TP
    FN = np.sum(conf_matrix, axis = 1) - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score) 
    
    log_path = f"master_thesis/results/evaluation{xml_no}_no_weight.log"
    os.makedirs(os.path.dirname(f"master_thesis/results/evaluation_log{xml_no}.txt"), exist_ok=True)
    
    with open(log_path, "w") as log_file:
        log_file.write(f"precision: {precision}\n")
        log_file.write(f"recall: {recall}\n")
        log_file.write(f"f1_score: {f1_score}\n")
        log_file.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        
    print(f"precision{precision},recall{recall},f1-score{f1_score}")    
              
#リボン図を生成する関数(重みなし)
def gen_ribbon_plot_no_weight(prediction_dict, xml_no, mode):
    plt.rcParams.update({'font.size': 14}) 
    
    colors = ['grey', 'orange', 'yellow', 'blue', 'black', 'green', 'purple']
    
    frames = sorted(prediction_dict.keys())
    
    if mode == "no_weight":
        predicted_labels = [int(prediction_dict[frame]['label'][0]) if isinstance(prediction_dict[frame]['label'], np.ndarray) else int(prediction_dict[frame]['label']) for frame in frames]   
    elif mode == "weight":
        predicted_labels = [int(prediction_dict[frame]['no_weight'][0]) if isinstance(prediction_dict[frame]['no_weight'], np.ndarray) else int(prediction_dict[frame]['no_weight']) for frame in frames]
        
    second_labels = [int(prediction_dict[frame]['second_label']) + 1 for frame in frames]#バグ応急処置
    actual_labels = [int(prediction_dict[frame]['actual_label']) if prediction_dict[frame]['actual_label'] is not None else 0 for frame in frames]
    
    fig, axs = plt.subplots(3, 1, figsize= (15, 3), sharex = True)
    
    #予測クラスのリボン図の描画
    for i, frame in enumerate(frames):
        predicted_y = 0.5
        axs[0].add_patch(Rectangle((frame, predicted_y), 48, 1, color = colors[predicted_labels[i]]))
    
    #2番目に信頼度の高いクラスのリボン図の描画    
    for i, frame in enumerate(frames):
        second_y = 0.5
        axs[1].add_patch(Rectangle((frame, second_y), 48, 1, color = colors[second_labels[i]]))
    
    #実際のクラスのリボン図の描画    
    for i, frame in enumerate(frames):
        actual_y = 0.5
        axs[2].add_patch(Rectangle((frame, actual_y), 48, 1, color = colors[actual_labels[i]])) 
    
    #軸の設定    
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Predicted', fontsize=12, labelpad=20)

    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Second', fontsize=12, labelpad=20)
    
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Actual', fontsize=12, labelpad=20)
    axs[2].set_xlabel('Frame Number (Time)')  
    
    axs[0].set_xlim(min(frames), max(frames))         
    
    #y軸目盛の非表示設定    
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
    
#リボン図を生成する関数(重みあり)
def gen_ribbon_plot_weight(prediction_dict, xml_no):
    plt.rcParams.update({'font.size': 14}) 
    
    colors = ['grey', 'orange', 'yellow', 'blue', 'black', 'green', 'purple']
    
    frames = sorted(prediction_dict.keys())
    
    predicted_labels = [int(prediction_dict[frame]['label'][0]) if isinstance(prediction_dict[frame]['label'], np.ndarray) else int(prediction_dict[frame]['label']) for frame in frames]
    dominant_labels = [int(prediction_dict[frame]['dominant_label']) if prediction_dict[frame]['dominant_label'] is not None else 0 for frame in frames]#バグ応急処置
    actual_labels = [int(prediction_dict[frame]['actual_label']) if prediction_dict[frame]['actual_label'] is not None else 0 for frame in frames]
    
    fig, axs = plt.subplots(3, 1, figsize= (15, 3), sharex = True)
    
    #予測クラスのリボン図の描画
    for i, frame in enumerate(frames):
        predicted_y = 0.5
        axs[0].add_patch(Rectangle((frame, predicted_y), 48, 1, color = colors[predicted_labels[i]]))
    
    #キューの独占的なクラスのリボン図の描画    
    for i, frame in enumerate(frames):
        second_y = 0.5
        axs[1].add_patch(Rectangle((frame, second_y), 48, 1, color = colors[dominant_labels[i]]))
    
    #実際のクラスのリボン図の描画    
    for i, frame in enumerate(frames):
        actual_y = 0.5
        axs[2].add_patch(Rectangle((frame, actual_y), 48, 1, color = colors[actual_labels[i]])) 
    
    #軸の設定    
    axs[0].set_ylim(0, 1)
    axs[0].set_ylabel('Predicted', fontsize=12, labelpad=20)

    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('dominant', fontsize=12, labelpad=20)
    
    axs[2].set_ylim(0, 1)
    axs[2].set_ylabel('Actual', fontsize=12, labelpad=20)
    axs[2].set_xlabel('Frame Number (Time)')  
    
    axs[0].set_xlim(min(frames), max(frames))         
    
    #y軸目盛の非表示設定    
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
    
    plt.suptitle(f'Ribbon Plot_weight No.{xml_no}')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, hspace=0.05)
    plt.savefig(f'master_thesis/results/ribbon_plot_{xml_no}_weight.png', bbox_inches='tight', pad_inches=0)  
    
#折れ線グラフを描画する関数
def draw_graph(frame_show, prob_graph_weight):
    
    class_labels = ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6']
    colors = ['orange', 'yellow', 'blue', 'black', 'green', 'purple']
    
    # 9回分の確率ベクトルがなければグラフを描画せずにフレームを返す
    if len(prob_graph_weight) < 9:
        return frame_show

    # 直近9回分の確率ベクトルを取得
    recent_no_weight = prob_graph_weight[-9:]
    x = np.arange(9)  # 0から8の範囲のx軸データ
    
    # PlotlyのFigureを作成
    fig = go.Figure()

    # 各クラスの折れ線を追加
    for class_idx in range(6):
        no_weight_values = [vec[class_idx] for vec in recent_no_weight]
        fig.add_trace(go.Scatter(x=x, y=no_weight_values, mode='lines',
                                 name=class_labels[class_idx],
                                 line=dict(color=colors[class_idx])))

    # レイアウトの設定
    fig.update_layout(
        title="Confidence Over Last 9 Predictions",
        xaxis_title="Prediction Step",
        yaxis_title="Confidence",
        xaxis=dict(tickmode='array', tickvals=np.arange(9)),
        yaxis=dict(range=[0, 1]),
        font=dict(size=18)
    )

    # グラフを画像として保存
    graph_image = fig.to_image(format="png")

    # OpenCVで画像を読み込み、NumPy配列に変換
    graph_image = np.frombuffer(graph_image, dtype=np.uint8)
    graph_image = cv2.imdecode(graph_image, cv2.IMREAD_COLOR)

    # グラフ画像のリサイズ
    graph_image = cv2.resize(graph_image, (frame_show.shape[1] // 2, frame_show.shape[0] // 2))  
    graph_height, graph_width, _ = graph_image.shape
    
    alpha = 0.7
    y_offset = 30 
    
    # グラフをフレームに合成
    frame_show[frame_show.shape[0] - graph_height - y_offset:frame_show.shape[0] - y_offset, 0:graph_width] = cv2.addWeighted(
        frame_show[frame_show.shape[0] - graph_height - y_offset:frame_show.shape[0] - y_offset, 0:graph_width],
        1 - alpha,
        graph_image,
        alpha,
        0
    )
    
    return frame_show

#全体における折れ線グラフを描画する関数(重みなし)
def plot_confidence_graph_no_weight(prob_graph_no_weight, xml_no):
    
    plt.clf()#プロットのクリア
    plt.figure(figsize=(15, 3))
    colors = ['orange', 'yellow', 'blue', 'black', 'green', 'purple']
    phases = ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6']
    
    # フレームごとにプロット
    for idx, phase in enumerate(phases):
        confidence_values = [frame[idx] for frame in prob_graph_no_weight]
        plt.plot(confidence_values, label=phase, color=colors[idx])
    
    output_file = f"master_thesis/results/confidence_graph_no_weight_{xml_no}.png"

    # グラフの装飾
    plt.xlabel('Steps')
    plt.ylabel('Confidence')
    plt.title(f'Confidence over Frames_no_weight_{xml_no}')
    plt.grid(True)
    plt.xlim(0, len(prob_graph_no_weight))
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()#余白調整
    
    
    # 画像を保存
    plt.savefig(output_file, format='png')
    plt.close()

#全体における折れ線グラフを描画する関数(重みあり)
def plot_confidence_graph(prob_graph_weight, xml_no, mode):
    
    plt.clf()#プロットのクリア
    plt.figure(figsize=(15, 3))
    colors = ['orange', 'yellow', 'blue', 'black', 'green', 'purple']
    phases = ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5', 'Phase6']
    
    # フレームごとにプロット
    for idx, phase in enumerate(phases):
        confidence_values = [frame[idx] for frame in prob_graph_weight]
        plt.plot(confidence_values, label=phase, color=colors[idx])
    
    output_file = f"master_thesis/results/confidence_graph_{mode}_{xml_no}.png"

    # グラフの装飾
    plt.xlabel('Steps')
    plt.ylabel('Confidence')
    plt.title(f'Confidence over Frames_{mode}_{xml_no}')
    plt.grid(True)
    plt.xlim(0, len(prob_graph_weight))
    plt.ylim(0, 1)
    plt.legend(loc = "upper right", bbox_to_anchor = (1.15, 1))
    plt.tight_layout()#余白調整
    
    
    # 画像を保存
    plt.savefig(output_file, format='png')
    plt.close()
