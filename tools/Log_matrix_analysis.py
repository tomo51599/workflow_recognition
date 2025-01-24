import os
import numpy as np
import pandas as pd

# スクリプトのカレントディレクトリを変更
os.chdir("/home/master_thesis")

# ログファイルが格納されているディレクトリ
directory_path = "results/model_13/weight_log"
print(directory_path)
# クラス数
num_classes = 6

# 結果を格納する辞書
metrics = {"precision": [], "recall": [], "f1_score": []}

# 配列文字列をPythonリストとしてパースする関数
def parse_array(array_str):
    array_str = array_str.replace("[", "").replace("]", "").strip()  # []を除去
    return np.array([float(x) for x in array_str.split() if x])  # 空白で分割してfloat変換

# ディレクトリ内のログファイルを取得
log_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".log")]

# ログファイルの処理
for file_path in log_files:
    if not os.path.exists(file_path):
        print(f"ファイルが見つかりません: {file_path}")
        continue

    with open(file_path, "r") as file:
        for line in file:
            if "precision" in line:
                precision = parse_array(line.split(":")[1].strip())
                metrics["precision"].append(precision)
            elif "recall" in line:
                recall = parse_array(line.split(":")[1].strip())
                metrics["recall"].append(recall)
            elif "f1_score" in line:
                f1_score = parse_array(line.split(":")[1].strip())
                metrics["f1_score"].append(f1_score)

# 各指標ごとのクラス別の平均・最大・最小を計算して出力
results = {}
for metric, values in metrics.items():
    if values:
        values = np.array(values)  # shape: (num_files, num_classes)
        mean_per_class = np.mean(values, axis=0)
        min_per_class = np.min(values, axis=0)
        max_per_class = np.max(values, axis=0)
        range_per_class = max_per_class - min_per_class

        class_results = []
        for class_idx in range(num_classes):
            class_results.append({
                "平均": round(mean_per_class[class_idx], 3),
                "±範囲": round(range_per_class[class_idx] / 2, 3),
                "最大": round(max_per_class[class_idx], 3),
                "最小": round(min_per_class[class_idx], 3)
            })
        results[metric] = class_results

# 結果を整形して表示
for metric, class_results in results.items():
    df = pd.DataFrame(class_results, index=[f"クラス {i+1}" for i in range(num_classes)])
    print(f"--- {metric} ---")
    print(df)
    print("\n")
