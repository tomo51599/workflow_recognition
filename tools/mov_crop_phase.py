import os
import cv2
import xml.etree.ElementTree as ET

def get_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return tree, root

def get_video_info(root):
    video_name = root.find("videoName").text
    phases = []
    for phase in root.findall("phase"):
        phase_no = int(phase.find("phaseNo").text)
        first_frame = int(phase.find("firstFrameNo").text)
        last_frame = int(phase.find("lastFrameNo").text)
        phases.append((phase_no, first_frame, last_frame))
    return video_name, phases

def get_current_phase(frame_No, phases):
    for phase_no, first_frame, last_frame in phases:
        if first_frame <= frame_No <= last_frame:
            return phase_no
    return None

def get_last_saved_number(save_dir, new_phase):
    if not os.path.exists(save_dir):
        return 0
    files = os.listdir(save_dir)
    phase_files = [f for f in files if f.startswith(f"_phase{new_phase}_") and f.endswith('.mp4')]
    numbers = [int(f.split('_')[2].split('.')[0]) for f in phase_files]
    return max(numbers) if numbers else 0

def start_new_video(save_dir, fps, new_phase, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_counter = get_last_saved_number(save_dir, new_phase) + 1
    file_path = os.path.join(save_dir, f"_phase{new_phase}_{file_counter}.mp4")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    print(file_path)
    return out, file_path

# メイン処理を関数化
def process_videos(xml_number):
    xml_path = os.path.join("master_thesis", "data", "xml", f"{xml_number:03}.xml")
    tree, root = get_xml(xml_path)
    video_name, phases = get_video_info(root)

    video_path = os.path.join("master_thesis", "data", "video", video_name)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_No = 0
    current_phase = None
    video_writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        new_phase = get_current_phase(frame_No, phases)
        if new_phase != current_phase or (frame_No % 48 == 0 and video_writer is not None):
            if video_writer is not None:
                video_writer.release()
            if new_phase is not None:
                save_dir = os.path.join("master_thesis", "data", "cropped_video", f"phase{new_phase}")
                os.makedirs(save_dir, exist_ok=True)
                video_writer, file_path = start_new_video(save_dir, fps, new_phase, frame_width, frame_height)
            current_phase = new_phase
        if video_writer is not None:
            video_writer.write(frame)
        frame_No += 1

    if video_writer is not None:
        video_writer.release()
    cap.release()

# XMLファイルの範囲指定
for i in range(1, 21):  
    process_videos(i)

               
        
    
