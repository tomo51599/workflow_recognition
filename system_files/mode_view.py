import cv2, os

from system_files.video_common_process import get_current_phase, phase_message, draw_text_on_frame

#ウィンドウに表示
def cap_view(video_path, phases):
    
    frame_No = 0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if cv2.waitKey(delay) &0xFF == ord("q") or not ret:
            break
        
        current_phase = get_current_phase(frame_No, phases)
    
        if get_current_phase is not None:
            cv2.putText(frame,
                        f"phase:{current_phase},current_frame{frame_No}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0 ,0),
                        2
                        )
            
            phase_text = phase_message(current_phase)
            frame = draw_text_on_frame(frame, phase_text, (10, frame.shape[0] - 30),  os.path.join("master_thesis","system_files", "meiryo.ttc"), 20, (255, 0, 0))
            
        cv2.imshow("video", frame)
               
        frame_No += 1 
    
    cap.release()
    cv2.destroyAllWindows()
        
        
            