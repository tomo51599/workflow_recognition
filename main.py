from system_files.file_info import set_xml_No, set_mode

set_xml_No("040")
set_mode("no_weight")

"""
view:xmlからフェーズ情報を取得し表示
no_weight:R(2+1)Dネットワークに入力し推論
weight:ネットワークの出力に補正
"""