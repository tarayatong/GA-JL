import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils_map import get_coco_map, get_map
#from utils_map import *
from yolo import YOLO

if __name__ == "__main__":

    #map_out_path    = r'yolo3/yolov4/logs/yolo_Attn/114_134/test_result/epbest/134'
    id = 'best'
    model_path= 'logs/noise_IRSTD_2/blur/ep'+id+'.pth'
    map_out_path = 'logs/noise_IRSTD_2/blur/test_result/ep'+id
    # image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    image_ids = open("model_data/IRSTD_test.txt").read().strip().split()
    #data_root = r'/home/liujing/Code/data/SAITD/noise/possion/129/'
    #data_root = '/home/liujing/Code/data/SAITD/SAITD/126/JPEGImages/'
    data_root = r'/home/liujing/Code/data/IRSTD-ISNet/noise/blur/IRSTD-ISNet/IRSTD1k_Img/'
    #data_root = r'/home/liujing/Code/data/IRSTD-ISNet/IRSTD1k_Img/'
    #data_root = r'/home/liujing/Code/data/SIRST_lj/test/image/'
    #print("dataId:{}".format("135"))
    MINOVERLAP = 0.25
    iou_count =0.0
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/plane_classes.txt'
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x，mAP0.x的意义是什么请同学们百度一下。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样本。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #--------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.25
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，本代码计算的Recall和Precision代表的是当门限值为0.5（此处定义为score_threhold）时所对应的Recall和Precision值。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision值。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    #VOCdevkit_path  = 'VOCdevkit'

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou,model_path=model_path)
        print("Load model done.")

        print("Get predict result.")
        infer_times = []
        k = 0
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(image_id)
            #image       = Image.open(data_root+"JPEGImages/"+image_path+".png")
            image = Image.open(data_root + image_path + ".png")
            #print(data_root + image_path + ".png")
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".png"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path,infer_times,k)
            k=k+1
        print("Get predict result done.")
        
    '''if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(data_root+"Annotations/"+image_id+".xml").getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")'''

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
