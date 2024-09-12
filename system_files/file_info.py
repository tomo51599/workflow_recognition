import os
import xml.etree.ElementTree as ET

from system_files.mode_view import cap_view
from system_files.mode_view_no_weight import cap_view_no_weight

video_path = None
xml_no_global = None
phases = []

#xml番号を設定
def set_xml_No(xml_no):
    global xml_no_global
    xml_no_global = xml_no 
    
    xml_path = os.path.join("master_thesis", "data", "xml", f"{xml_no}.xml")    
    get_xml(xml_path)        

#xml読み込み
def get_xml(xml_path):
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    get_video_info(root)

#情報を取得,ビデオパスとフェーズリストを返す
def get_video_info(root):
    global video_path, phases
    
    video_name = root.find("videoName").text
    phases = []
    
    for phase in root.findall("phase"):
        phase_no = int(phase.find("phaseNo").text)
        first_frame = int(phase.find("firstFrameNo").text)
        last_frame = int(phase.find("lastFrameNo").text)
        phases.append((phase_no, first_frame, last_frame))
        
    video_path = os.path.join("master_thesis","data","video",video_name) 
    
#モードセット
def set_mode(mode):
    if mode == "view":
        cap_view(video_path, phases)
    elif mode == "no_weight":
        cap_view_no_weight(video_path, phases, xml_no_global, mode)
    elif mode == "weight":
        print("3")
    else:
        print("error")             
         