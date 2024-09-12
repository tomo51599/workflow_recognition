from system_files.file_info import set_xml_No, set_mode

xml_no = "040"
mode = "no_weight"

set_xml_No(xml_no)
set_mode(mode)

"""
view:xmlからフェーズ情報を取得し表示
no_weight:R(2+1)Dネットワークに入力し推論
weight:ネットワークの出力に補正
"""