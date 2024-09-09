import keras
from keras import layers
import einops
from keras.layers import Layer
from keras.utils import register_keras_serializable
import os 
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import pathlib
import random
import collections
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

#モデルのロード##################################################
# Define the dimensions of one frame in the set of frames created
HEIGHT = 224
WIDTH = 224

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()


@keras.saving.register_keras_serializable(package="MyCustomLayers")
class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding, **kwargs):
        super().__init__(**kwargs)  # 親クラスのコンストラクタを呼び出す
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seq = keras.Sequential([
            # Spatial decomposition
            keras.layers.Conv3D(filters=self.filters,
                                kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
                                padding=self.padding),
            # Temporal decomposition
            keras.layers.Conv3D(filters=self.filters,
                                kernel_size=(self.kernel_size[0], 1, 1),
                                padding=self.padding)
        ])
  
    def call(self, x):
        return self.seq(x)
    
    def get_config(self):
        config = super().get_config()  # 親クラスの設定を取得
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'padding': self.padding
        })
        return config

@keras.saving.register_keras_serializable(package="CustomLayers")
class ResidualMain(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualMain, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])
  
    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super(ResidualMain, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

@keras.saving.register_keras_serializable(package="CustomLayers")
class Project(Layer):
    def __init__(self, units, **kwargs):
        super(Project, self).__init__(**kwargs)
        self.units = units
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

    def get_config(self):
        config = super(Project, self).get_config()
        config.update({'units': self.units})
        return config

@keras.saving.register_keras_serializable(package="my_package", name="custom_fn")
def add_residual_block(input, filters, kernel_size):

  out = ResidualMain(filters, 
                     kernel_size)(input)
  
  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])

@keras.saving.register_keras_serializable(package="CustomLayers")
class ResizeVideo(Layer):
    def __init__(self, height, width, **kwargs):
        super(ResizeVideo, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        # b stands for batch size, t stands for time, h stands for height, 
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos

    def get_config(self):
        config = super(ResizeVideo, self).get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config
  
input_shape = (None, 48, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Flatten()(x)
x = layers.Dense(6)(x)
#layers.Dense(6, activation='softmax')(x)

custom_objects = {
    "Conv2Plus1D": Conv2Plus1D,
    "ResidualMain": ResidualMain,
    "Project": Project,
    "ResizeVideo": ResizeVideo,
    "add_residual_block": add_residual_block  
}

model = keras.models.load_model("master_thesis/system_files/test_model_9.keras", custom_objects=custom_objects)   
#モデルのロード##################################################


#ビデオの表示####################################################
xml_no = "040"
xml_path = os.path.join("master_thesis", "data", "xml", f"{xml_no}.xml")
print(xml_path)

#xml読み込み
def get_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    return tree, root

#情報を取得
def get_video_info(root):
    video_name = root.find("videoName").text
    phases = []
    
    for phase in root.findall("phase"):
        phase_no = int(phase.find("phaseNo").text)
        first_frame = int(phase.find("firstFrameNo").text)
        last_frame = int(phase.find("lastFrameNo").text)
        phases.append((phase_no, first_frame, last_frame))
    
    return video_name, phases

#現在のフェーズを決定する関数
def get_current_phase(frame_No, phases):
    
    for phase_no, first_frame, last_frame in phases:
        if first_frame <= frame_No <= last_frame:
            
            return phase_no
      
    return None

#ファイル名の決定関数
def get_last_saved_number(save_dir):
    if not os.path.exists(save_dir):
        return 0
    files = os.listdir(save_dir)
    numbers = [int(f.split('.')[0]) for f in files if f.endswith('.mp4')]
    if not numbers:
        return 0
    return max(numbers)    

#ビデオのクロップ
def start_new_video(save_dir, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ファイルフォーマット
    file_counter = get_last_saved_number(save_dir) + 1  # 更新された関数を使用
    file_path = os.path.join(save_dir, f"{file_counter}.mp4")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
    file_path = os.path.join(save_dir, f"{file_counter - 1}.mp4")
    return out, file_path

#フレーム処理
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
            # TensorFlowの操作を使ってゼロフレームを作成
            result.append(tf.zeros_like(result[0]))
    src.release()

    # TensorFlow操作を使用しているため、結果をnumpy配列に変換
    result = tf.stack(result).numpy()
    return result

class FrameGenerator:
  
  def __init__(self, path, n_frames, training=False):
    # pathがディレクトリかファイルかをチェック
    if os.path.isdir(path):
      self.path = pathlib.Path(path)
      self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
      self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))
    elif os.path.isfile(path):
      self.path = [pathlib.Path(path)]  # リストとして保持
      self.class_names = ['single_file']
      self.class_ids_for_name = {'single_file': 0}
    #else:
      #raise ValueError(f"Provided path {path} is neither a directory nor a file.")
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
      
# 推論を実行する関数
def get_actual_predicted_labels(dataset): 

  predicted = model.predict(dataset)
  predict_probability = predicted
  
  predicted = tf.concat(predicted, axis=0)
  predicted = tf.argmax(predicted, axis=1)

  return predicted, predict_probability

#1番目に信頼度の高いクラスの確率値を取得
def get_first_predict_labels(predict_probability):
    
    probabilities = np.array(predict_probability)
    sorted_indices = np.argsort(probabilities)[::-1]
    first_largest_index = sorted_indices[0]
    first_largest_probability = probabilities[first_largest_index]
    
    return  first_largest_probability

#2番目に信頼度の高いクラスの確率値とクラス名を取得
def get_second_predict_labels(predict_probability, class_names):
    
    probabilities = np.array(predict_probability)
    sorted_indices = np.argsort(probabilities)[::-1]
    second_largest_index = sorted_indices[1]
    second_largest_probability = probabilities[second_largest_index]
    second_largest_class = class_names[second_largest_index]
    
    return second_largest_class, second_largest_probability

#真値の確率値を取得
def get_current_phase_probability(predict_probability, current_phase):
   
   if current_phase is not None: 
        current_phase_cal = current_phase - 1
        current_phase_probability =  predict_probability[current_phase_cal]
        
        return current_phase_probability
    
   else:
       return None

#推定値のカウント
def count_prediction(predicted_phase_No, current_phase, results):
   
    if current_phase is not None:
        results[current_phase][predicted_phase_No] += 1
    
    return results

# リボン図の生成関数
def generate_ribbon_plot(predict_results, xml_no):
    fig, ax = plt.subplots(figsize=(15, 5))
    colors = ['orange', 'yellow', 'blue', 'black', 'green', 'purple']
    phases = sorted(predict_results.keys())
    for phase in phases:
        for frame_No, predicted_phase in predict_results[phase]:
            predicted_phase = int(predicted_phase)
            ax.scatter(frame_No, phase, c=colors[predicted_phase-1], label=f'Phase {predicted_phase}', s=10)
    ax.set_xlabel('Time (Frame Number)')
    ax.set_ylabel('Phase')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1), title="Legend")
    plt.title(f'Phase Recognition Ribbon Plot for Test_{xml_no}')
    plt.tight_layout()
    plt.savefig(f'/home/master_thesis/ribbon_plot_{xml_no}_No_weight.png') 
               
# メイン処理
tree, root = get_xml(xml_path)
video_name, phases = get_video_info(root)

video_path = os.path.join("master_thesis", "data", "video", video_name)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_No = 0
weight = 0
current_phase = None
video_writer = None

n_frames = 24
batch_size = 1

predict_results = defaultdict(list)

predicted_labels = None
determined_phase = None
dominant_phase = None

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

class_names = ['phase1', 'phase2', 'phase3', 'phase4', 'phase5', 'phase6']



class_name_to_index = {name: idx for idx, name in enumerate(class_names)}
results = {phase: {predicted_phase: 0 for predicted_phase in range(1, 7)} for phase in range(1, 7)}

while cap.isOpened():
    ret, frame = cap.read()
    
    if frame is not None:
        frame_2 = frame.copy()
    
    if cv2.waitKey(5) & 0xFF == ord("q") or not ret:
        break
             
    if frame_No != 0 and frame_No % 48 == 0:
        
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        save_dir = os.path.join("master_thesis", "data", "visualize_evaluation")
        os.makedirs(save_dir, exist_ok=True)
        video_writer, file_path = start_new_video(save_dir, fps, frame_width, frame_height)
        
    if video_writer is not None:
        video_writer.write(frame)
            
    if frame_No > 49 and frame_No % 48 == 0:
        test_ds = tf.data.Dataset.from_generator(FrameGenerator(file_path, n_frames), output_signature = output_signature) 
        test_ds = test_ds.batch(batch_size)

        
        fg = FrameGenerator(file_path, n_frames, training=False) 
        predicted, predict_probability = get_actual_predicted_labels(test_ds)
        
        predicted_indices = predicted  
        predicted_labels = [class_names[idx] for idx in predicted_indices.numpy()]
        predicted_labels = [int(idx.numpy() + 1) for idx in predicted_indices]
        
        predict_count = [class_name_to_index[p] + 1 for p in predicted_labels]

        if current_phase is not None:
            for pc in predict_count:
                results = count_prediction(pc, current_phase, results)
            
            for predicted_phase in predicted_labels:
                predict_results[current_phase].append((frame_No, predicted_phase))    
        
      

        first_largest_probability = get_first_predict_labels(predict_probability[0])
        second_class, second_largest_probability = get_second_predict_labels(predict_probability[0] , class_names)
        current_class_probability = get_current_phase_probability(predict_probability[0], current_phase)
        rounded_probability = np.around(np.array(predict_probability[0]), decimals = 3)
                                
        
        current_phase = get_current_phase(frame_No, phases)
    
    
    if predicted_labels is not None:
        cv2.putText(frame_2,
                        f"actual_class:['phase{current_phase}'] predicted_class{predicted_labels} second_class:['{second_class}'] frame{frame_No}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2
                        )
        
        cv2.putText(frame_2,
                        f"acutual_class:{'N/A' if current_class_probability is None else f'{current_class_probability:.2f}'}, predict_class:{first_largest_probability:.2f}, second_class:{second_largest_probability:.2f}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2
                        )
        
        cv2.putText(frame_2,
                        f"score:{rounded_probability} ",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2
                        )
   


            
    cv2.imshow("video", frame_2) 
            
    frame_No += 1
 
if video_writer is not None:
    video_writer.release()

print(results)

size = len(results)
conf_matrix = np.zeros((size, size))

for actual, predicted_counts in results.items():
    for predicted, count in predicted_counts.items():
        conf_matrix[actual-1, predicted-1] = count
        
conf_matrix = conf_matrix.astype(int)

plt.figure(figsize = (10, 8))
sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap= "viridis", xticklabels=[f'phase{i}' for i in range(1, size+1)], yticklabels=[f'phase{i}' for i in range(1, size+1)])
plt.ylabel('Actual Phase')
plt.title(f'Confusion Matrix of Action Recognition for Test_{xml_no}_No_weight')
plt.savefig(f'/home/master_thesis/confusion_matrix_{xml_no}_No_weight.png')    

cap.release()
cv2.destroyAllWindows()

generate_ribbon_plot(predict_results, xml_no)