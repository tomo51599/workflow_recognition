# モデルのパス
model_path = "master_thesis/system_files/test_model_9.keras"

#evaluation_visualize用のビデオ保存パス
save_dir = "master_thesis/data/visualize_evaluation"

#重みを定義 
def cal_weight(weight):

    weight_list = [0] * 6
    
    if weight == 0:
        weight_list = [1, 0, 0, 0, 0, 0]
    elif weight == 1:
        weight_list = [1, 1, 0, 0, 0, 0]
    elif weight == 2:
        weight_list = [0, 1, 1, 0, 0, 0]
    elif weight == 3:
        weight_list = [0, 0, 1, 1, 0, 0]
    elif weight == 4:
        weight_list = [0, 0, 0, 1, 1, 0]
    elif weight == 5:
        weight_list = [0, 0, 0, 0, 1, 1]
    elif weight == 6:
        weight_list = [0, 0, 0, 0, 0, 1]
    else:
        weight_list = [1] * 6

    return weight_list

class_names = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']

    
    
