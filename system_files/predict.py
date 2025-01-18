import keras, collections
import numpy as np
import tensorflow as tf

from system_files.custom_layer import *
from system_files.config_file import *
from system_files.video_common_process import *

# 推論を実行する関数
def get_actual_predicted_labels(dataset): 
  
  predicted = model.predict(dataset)
  
  predicted_label = (tf.argmax(predicted, axis=1).numpy() + 1)
  predicted = np.around(np.array(predicted[0]), decimals = 3)
  
  return predicted, predicted_label

#真値の確率値を取得
def get_actual_label_prob(predicted, current_phase):
   
   if current_phase is not None: 
        current_phase_cal = current_phase - 1
        actual_label_prob =  predicted[current_phase_cal]
        
        return actual_label_prob
    
   else:
       return None

#1番目に信頼度の高いクラスの確率値を取得
def get_first_predict_labels(predicted):
    
    probabilities = np.array(predicted)
    sorted_indices = np.argsort(probabilities)[::-1]
    first_largest_index = sorted_indices[0]
    first_largest_prob = np.around(probabilities[first_largest_index], decimals = 3)
    
    return  first_largest_prob

#2番目に信頼度の高いクラスの確率値とクラス名を取得
def get_second_predict_labels(predicted):
    
    probabilities = np.array(predicted)
    sorted_indices = np.argsort(probabilities)[::-1]
    second_largest_class = sorted_indices[1]
    second_largest_prob = np.around(probabilities[second_largest_class], decimals = 3)
    
    return second_largest_class, second_largest_prob

#支配的な推論結果を計算
def determine_dominant_phase(predicted_list):
    
    flat_list = [item for sublist in predicted_list for item in sublist]
    phase_counter = collections.Counter(flat_list)
    dominant_phase, count = phase_counter.most_common(1)[0]
    
    if count >= 9 // 2 +1:
        return dominant_phase
    else:
        return None 

#重みがけを実行    
def apply_weights_to_prediction(predict_probabilities, weight):

    weight_list = cal_weight(weight)
    
    # 重みを確率に適用
    adjusted_probabilities = predict_probabilities * np.array(weight_list)
    
    sum_probabilities = np.sum(adjusted_probabilities, keepdims = True)
    print(sum_probabilities)
    adjusted_probabilities = adjusted_probabilities / sum_probabilities
    
    return adjusted_probabilities, weight_list

#最終的な推論結果を出力
def determine_adjusted_prediction(adjusted_probabilities):
   
    class_names = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']
   
    if adjusted_probabilities.ndim == 1:
        adjusted_probabilities = np.atleast_2d(adjusted_probabilities)
   
    adjusted_prediction_indices = np.argmax(adjusted_probabilities, axis=1)
    adjusted_class_label = [class_names[idx] for idx in adjusted_prediction_indices]
   
    return adjusted_prediction_indices +1, adjusted_class_label


#モデルのロード
model = keras.models.load_model(model_path, custom_objects={
    "Conv2Plus1D": Conv2Plus1D,
    "ResidualMain": ResidualMain,
    "Project": Project,
    "ResizeVideo": ResizeVideo,
    "add_residual_block": add_residual_block
})