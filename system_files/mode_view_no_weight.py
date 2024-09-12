import os, cv2
import tensorflow as tf
import concurrent.futures

from queue import Queue
from system_files.predict import*
from system_files.video_common_process import*
from system_files.result import*

#動画メイン処理            
def cap_view_no_weight(video_path, phases, xml_no, mode):
    
    frame_No = 0 
    frame_No_show = 0  
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
    exit_flag = False #終了処理
    
    prediction_dict = {}
    last_prediction = None
    
    results = {phase: {predicted_phase: 0 for predicted_phase in range(1, 7)} for phase in range(1, 7)}

    while cap.isOpened() and not exit_flag:  
        ret, frame = cap.read()
        frame_queue.put(frame)
        current_phase = get_current_phase(frame_No_show, phases)
       
        if not ret or frame is None:
            exit_flag = True
            break
       
        if  cv2.waitKey(5) & 0xFF == ord("q"):
            exit_flag = True 
        
        if frame_queue.qsize() > 48 and frame_No % 48 == 0:
           future = executor.submit(process_video_writer, frame_No, save_dir, fps, frame_width, frame_height, frame_queue)
           file_path = future.result()
           
           test_ds = tf.data.Dataset.from_generator(FrameGenerator(file_path, n_frames), output_signature = output_signature) 
           test_ds = test_ds.batch(batch_size)
           
           predicted, predicted_label= get_actual_predicted_labels(test_ds)
           
           prediction_dict[frame_No] = {'confidence': predicted, 'label': predicted_label, 'actual': current_phase}
           last_prediction = {'confidence': predicted, 'label': predicted_label} 
           print(predicted_label)
           
           for predicted_phase in predicted_label:
                results  = count_prediction(predicted_phase, current_phase, results)

        #フレームバッファ関連処理
        if len(buffered_frames) < 96:
            buffered_frames.append(frame)

        if len(buffered_frames) >= 96:
            frame_show = buffered_frames.pop(0)
            
            #日本語の描画
            phase_text = phase_message(current_phase)
            frame_show = draw_text_on_frame(frame, phase_text, (10, frame.shape[0] - 30),  os.path.join("master_thesis","system_files", "meiryo.ttc"), 20, (255, 0, 0))
            
            if last_prediction:
                prediction_text = f"prediction:phase{last_prediction['label']} {last_prediction['confidence']}"
                cv2.putText(frame_show,f"current:phase:{current_phase},frame{frame_No_show}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 2)
                cv2.putText(frame_show, prediction_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("video", frame_show)
            
            frame_No_show += 1
                 
        frame_No += 1

    executor.shutdown(wait=True)  
    print(results)
    gen_conf_matrix(results, xml_no, mode)

    cv2.destroyAllWindows() 
    cap.release()  
 
    