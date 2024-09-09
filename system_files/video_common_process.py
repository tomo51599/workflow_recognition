import cv2, random, os, pathlib
import tensorflow as tf
import numpy as np

from PIL import ImageFont, ImageDraw, Image

#メッセージ関数
def phase_message(current_phase):
    
    if current_phase is None:
        return "フェーズ情報なし"
    
    cal_message = current_phase - 1
    
    message = ["1_皮膚の切開（開腹処置)_メスが見えてから剪刀が見えなくなるまで",
               "2_子宮の吊り出し_フックまたはピンセットが見えてから1つ目の卵巣を鉗子で挟む前まで",
               "3_1つめの卵巣の血管処理_鉗子が見えてから処理を終えた剪刀が見えなくなるまで",
               "4_2つめの卵巣の血管処理_前のフレームからを終えた剪刀が見えなくなるまで",
               "5_子宮の根元の血管処理・子宮摘出_前のフレームから子宮の根元を押し込んで見えなくなるまで",
               "6_皮膚の縫合（閉腹処置)_持針器またはピンセットが入ってきてから剪刀が見えなくなるまで"]
    
    if 0 <= cal_message < len(message):
        load_message = message[cal_message]
        return load_message
  
#日本語描画
def draw_text_on_frame(frame, text, position, font_path, font_size, color):
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    font = ImageFont.truetype(font_path, font_size)

    draw.text(position, text, font=font, fill=color)

    frame_np = np.array(frame_pil)
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    return frame_np

# 現在のフェーズを決定する関数
def get_current_phase(frame_No, phases):
   
    for phase_no, first_frame, last_frame in phases:
        if first_frame <= frame_No <= last_frame:
            return phase_no
   
    return None

# ファイル名の決定関数
def get_last_saved_number(save_dir):
   
    if not os.path.exists(save_dir):
        return 0
    files = os.listdir(save_dir)
    numbers = [int(f.split('.')[0]) for f in files if f.endswith('.mp4')]
   
    if not numbers:
        return 0
   
    return max(numbers) 

# ビデオのクロップ
def write_video(frame_No, video_writer, save_dir, fps, frame_width, frame_height, frame):
    file_path = None
    
    if frame_No != 0 and frame_No % 48 == 0:
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            
        video_writer, file_path = crop_video(save_dir, fps, frame_width, frame_height)
           
    if video_writer is not None:
        video_writer.write(frame) 
        
    return video_writer, file_path       

def crop_video(save_dir, fps, width, height):
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    file_counter = get_last_saved_number(save_dir) + 1  
    file_path = os.path.join(save_dir, f"{file_counter}.mp4")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    
    return out, file_path

#遅延処理
def process_video_writer(frame_No, save_dir, fps, frame_width, frame_height, frame_queue):
    video_writer = None
    frames_to_process = []
    file_path = None  

    while not frame_queue.empty() and len(frames_to_process) < 48:
        frames_to_process.append(frame_queue.get())
        frame_queue.task_done()

    for frame in frames_to_process:
        video_writer, temp_file_path = write_video(frame_No, video_writer, save_dir, fps, frame_width, frame_height, frame)
        
        if temp_file_path:
            file_path = temp_file_path
        
        frame_No += 1

    if video_writer is not None:
        video_writer.release()

    return file_path

# フレーム処理
def format_frames(frame, output_size):
    
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=2):
    
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCVから読み込まれたフレームをRGBに変換
        frame = format_frames(frame, output_size)
        result.append(frame)

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # RGB変換
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(tf.zeros_like(result[0]))
    src.release()

    result = tf.stack(result).numpy()
    return result

class FrameGenerator:
    
    def __init__(self, path, n_frames, training=False):
        if os.path.isdir(path):
            self.path = pathlib.Path(path)
            self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
            self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
        elif os.path.isfile(path):
            self.path = [pathlib.Path(path)]  # リストとして保持
            self.class_names = ['single_file']
            self.class_ids_for_name = {'single_file': 0}
        self.n_frames = n_frames
        self.training = training

    def get_files_and_class_names(self):
    
        if 'single_file' in self.class_names:
            return self.path, self.class_names  # 単一ファイルの場合
        else:
            video_paths = list(self.path.glob('*/*.mp4'))
            classes = [p.parent.name for p in video_paths]
            return video_paths, classes

    def __call__(self):
    
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        if self.training:
            random.shuffle(pairs)
        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, label