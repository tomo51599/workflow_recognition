import os, cv2
import tensorflow as tf
import concurrent.futures

from queue import Queue

from system_files.predict import*
from system_files.video_common_process import*
from system_files.result import*

#支配的な推論結果を計算
def determine_dominant_phase(predicted_list):
    flat_list = [item for sublist in predicted_list for item in sublist]
    phase_counter = collections.Counter(flat_list)
    dominant_phase, count = phase_counter.most_common(1)[0]
    
    if count >= 9 // 2 +1:
        return dominant_phase
    else:
        return None 

#重みを定義 
def cal_weight(weight):

    weight_list = [0] * 6
    
    if weight == 0:
        weight_list[0] = 1
    elif weight == 1:
        weight_list[:2] = [1, 1]
    elif weight == 2:
        weight_list[:3] = [0, 1, 1]
    elif weight == 3:
        weight_list[1:4] = [0, 1, 1]
    elif weight == 4:
        weight_list[2:5] = [0, 1, 1]
    elif weight == 5:
        weight_list[3:6] = [0, 1, 1]
    elif weight == 6:
        weight_list[4:6] = [0, 1]
    else:
        weight_list = [1] * 6

    return weight_list

#重みがけを実行    
def apply_weights_to_prediction(predict_probabilities, weight):

    weight_list = cal_weight(weight)
    
    adjusted_probabilities = predict_probabilities * np.array(weight_list)#重みを確率に適用
    
    sum_probabilities = np.sum(adjusted_probabilities, keepdims = True)
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

#推定値のカウント
def count_prediction(predicted_phase_No, current_phase, results):
   
    if current_phase is not None:
        results[current_phase][predicted_phase_No] += 1
    
    return results

#動画メイン処理            
def cap_view_weight(video_path, phases, xml_no, mode):
    
    frame_No = 0 
    frame_No_show = 0  
    weight = 0
    n_frames = 24
    batch_size = 1
      
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay = int(1000 / fps)
    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

    save_dir = os.path.join("master_thesis", "data", "visualize_evaluation")
    os.makedirs(save_dir, exist_ok=True)

    frame_queue = Queue(maxsize=96) 
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    buffered_frames = [] 
    prediction_buffer = []
    predicted_list =[]
    frame_phase_changed = []
    prob_graph_weight = []
     
    determined_phase = None
    dominant_phase = None
    dominant_phase_dic = None
    
    prediction_dict = {}
    exit_flag = False #終了処理
    
    results = {phase: {predicted_phase: 0 for predicted_phase in range(1, 7)} for phase in range(1, 7)}
    
    while cap.isOpened() and not exit_flag:  
        ret, frame = cap.read()
        frame_queue.put(frame)
        current_phase = get_current_phase(frame_No_show, phases)
       
        #終了処理
        if not ret or frame is None:
            exit_flag = True
            break
       
        if  cv2.waitKey(5) & 0xFF == ord("q"):
            exit_flag = True 
        
        #48フレーム目以降で推論を開始
        if frame_queue.qsize() > 48 and frame_No % 48 == 0:
           future = executor.submit(process_video_writer, frame_No, save_dir, fps, frame_width, frame_height, frame_queue)
           file_path = future.result()#バッファ関連処理
           
           test_ds = tf.data.Dataset.from_generator(FrameGenerator(file_path, n_frames), output_signature = output_signature) #モデルへ入力
           test_ds = test_ds.batch(batch_size)
           
           predicted, predicted_label= get_actual_predicted_labels(test_ds)#推論を実行、確率ベクトルと推論したラベルを取得
           actual_label_prob = get_actual_label_prob(predicted, current_phase)#真値の信頼度を取得　ベクトル
           first_largest_prob = get_first_predict_labels(predicted)#最も信頼度の高いクラスの信頼度を取得　float型数値
           second_largest_class, second_largest_prob = get_second_predict_labels(predicted)#2番目に信頼度が高いクラスと信頼度を取得　float型数値
           
           adjusted_probabilities, weight_list = apply_weights_to_prediction(predicted, weight)#確率ベクトルに重みを適用 ベクトル
           adjusted_predictions, adjusted_class_label = determine_adjusted_prediction(adjusted_probabilities)#最終推論を決定　クラス番号とクラスラベル
           predicted_list.append(adjusted_class_label)#支配的なフェーズを判定するリストに保存
           
           prob_graph_weight.append(adjusted_probabilities)
           
           if len(predicted_list) > 9:
            predicted_list.pop(0)#キューの要素が9個以上でdeque
            dominant_phase = determine_dominant_phase(predicted_list)
            
            #重みの更新
            if determined_phase != dominant_phase and not None and dominant_phase is not None:
                predicted_list.clear()
                weight += 1
                print(weight)
                frame_phase_changed.append(frame_No)
                print(frame_phase_changed)
            
            if dominant_phase is not None:
                determined_phase = dominant_phase
           
           last_prediction = {'frame_No': frame_No, 'frame_No_show':frame_No_show, 
                              'confidence': predicted, 'label': predicted_label, 
                              'actual_label': current_phase,'actual_prob': actual_label_prob,
                              'second_label': second_largest_class, 'second_largest_prob': second_largest_prob}
           prediction_buffer.append(last_prediction) 
           
           for predicted_phase in adjusted_predictions:
                results  = count_prediction(predicted_phase, current_phase, results) #結果のカウント
                
                if dominant_phase is not None:
                    dominant_phase_dic = int(dominant_phase[5])
                
                prediction_dict[frame_No] = {'confidence': predicted, 'label': predicted_phase, 'prob': first_largest_prob, 
                                        'actual_label': current_phase, 'actual_label_prob': actual_label_prob,
                                        'dominant_label': dominant_phase_dic, 'second_largest_prob': second_largest_prob}

        #フレームバッファ関連処理
        if len(buffered_frames) < 97:
            buffered_frames.append(frame) 

        if len(buffered_frames) >= 97:
            frame_show = buffered_frames.pop(0)
            prediction_show = None
            
            if len(prediction_buffer) >= 3:
                prediction_show = prediction_buffer.pop(0)
                
            else:
                prediction_show = last_prediction
                
            #日本語の描画
            phase_text = phase_message(current_phase)
            frame_show = draw_text_on_frame(frame, phase_text, (10, frame.shape[0] - 30),  os.path.join("master_thesis","system_files", "meiryo.ttc"), 20, (255, 0, 0))
            
            #判定結果の描画
            if prediction_show:
                prediction_text = f"prediction:phase{prediction_show['label']} {prediction_show['confidence']}"
                adjusted_text = f"adjusted:phase{adjusted_class_label}{np.around(adjusted_probabilities, decimals = 3)}"
                weight_text = f"weight:{weight_list}"
                prediction_list_text = f"prediction_list{predicted_list}"
                dominant_phase_text = f"dominant_class:{dominant_phase}"
                cv2.putText(frame_show,f"current:phase:{current_phase},frame{frame_No_show}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 2)
                cv2.putText(frame_show, prediction_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame_show, adjusted_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame_show, weight_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame_show, prediction_list_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame_show, dominant_phase_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
            frame_show = draw_graph(frame_show, prob_graph_weight)#左下に補正信頼度の折れ線グラフの描画※実行速度への影響大    

            cv2.imshow("video", frame_show)
            
            frame_No_show += 1
                 
        frame_No += 1

    executor.shutdown(wait=True)  
    gen_conf_matrix(results, xml_no, mode)
    gen_ribbon_plot_weight(prediction_dict, xml_no)

    cv2.destroyAllWindows() 
    cap.release()  
 
    