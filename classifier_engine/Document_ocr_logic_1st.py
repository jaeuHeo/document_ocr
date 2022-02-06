#!/usr/bin/env python
# coding: utf-8

# version 확인
# torch                  1.7.0
# torchaudio             0.7.2
# torchtext              0.6.0
# torchvision            0.8.1

# # Text dection - OCR - KIE
# 
# 하나의 이미지를 위의 프로세스를 거쳐 결과 도출
# , 각각의 단계별 도출값 확인 가능

# In[ ]:


get_ipython().system(' pip3 list | grep torch')


# In[ ]:


## Text dection


# In[130]:


import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from text_detection import craft_utils
from text_detection import imgproc
import text_detection.file_utils
import json
import zipfile

from text_detection.craft import CRAFT

from collections import OrderedDict

import matplotlib.pyplot as plt

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly,refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()


    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    
    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# ### 실행

# In[157]:


class Args():
    def __init__(self,cuda=False, trained_model='weights/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size =1280, mag_ratio=1.5, poly=False, show_time=False,test_folder='/data/',refine=False, refiner_model='weights/craft_refiner_CTW1500.pth'):
        self.cuda = cuda
        self.trained_model = trained_model = trained_model
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.show_time = show_time
        self.test_folder = test_folder
        self.refine = refine
        self.refiner_model = refiner_model
        
        
def img_show(img, size =(15,15)):
    plt.rcParams["figure.figsize"] = size
    imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    


# #### 이미지 입력: 폴더 단위

# In[158]:


import os

label_path_list = [
                            ['0', '사업자등록증'],
#                             ['1', '토지대장'],
#                             ['2', '특허증'],
#                             ['3', '명함'],
#                             ['4', '상장'],
#                             ['5', '거래명세서'],
#                             ['6', '사원증'],
#                             ['7', '부동산매매계약서'],
#                             ['8', '사업자등록증명원'],
#                             ['9', '미등록문서'],
                            
                                                    ]

data_dir ='./scan_test/'
image_list = []
t = time.time()
for file_idx, _ in label_path_list:

    path2data = os.path.join(data_dir, file_idx)
    filenames = os.listdir(path2data)
    if '.ipynb_checkpoints' in filenames:
        
        filenames.remove('.ipynb_checkpoints')
        
    for c, i in enumerate(filenames):
        ######사업자등록증#######
        if i == 'IMG_7602.jpg':
            image_list.insert(0,os.path.join(path2data, i))
        elif i == 'IMG_7599.jpg':
            image_list.append(os.path.join(path2data, i))
        ########토지대장########3
#         if i == 'IMG_7617.jpg':
#             image_list.insert(0,os.path.join(path2data, i))
#         elif i == 'IMG_7616.jpg':
#             image_list.append(os.path.join(path2data, i))
        ##########특허증##########
#         if i == 'certif_01-1.png':
#             image_list.insert(0,os.path.join(path2data, i))
#         elif i == 'IMG_7618.jpg':
#             image_list.append(os.path.join(path2data, i))
    
args = Args(test_folder='./scan_test/',text_threshold=0.8,link_threshold=0.4,canvas_size=1800,refine=False,poly=False,cuda=True)
t_1 = time.time() - t
print(f"time : {t_1}s")
labels=[(int(i.split('/')[-2])) for i in image_list]
# labels


# In[159]:


image_list


# In[160]:


import csv
import re
bbox_list={
           'value': [],
           'value_label' : []}
# saubja_regist.csv
# toji_regist.csv
image = cv2.imread(image_list[0], cv2.IMREAD_COLOR)
# image = cv2.resize(image, dsize=(3000, 4000))
with open('saubja_val.csv', 'rU') as p:
    
    for idx,ii in enumerate(csv.reader(p,delimiter=',')):
#         print(ii)
        key_li = []
        val_li = []
        for i in range(0,len(ii)):
            
            ii[i] = ii[i].replace("[",'').replace("]",'')
            ii[i] = re.split('[,]',ii[i])
        for k in range(len(ii)):
            for l in range(len(ii[k])):
#                 print(ii[k][l])
                ii[k][l] = float(ii[k][l])
#         print(ii)
        img = image.copy()
#         if idx % 2 !=0 or idx==0:
# #             img = image.copy()
#             pts = np.array(ii).reshape(-1,2).astype(np.int32)

#             img = cv2.polylines(img, [pts], True, (0,0,255),2)
#             img_show(img)
#             print(ii)
#             bbox_list['key'].append(ii)
#             bbox_list['key_label'].append(input('키 라벨: '))
#         else:
#             img = image.copy()
        pts = np.array(ii).reshape(-1,2).astype(np.int32)

        img = cv2.polylines(img, [pts], True, (0,0,255),2)

        img_show(img)
        print(ii)
        bbox_list['value'].append(ii)
        bbox_list['value_label'].append(input('밸류 라벨: '))


bbox_list


# In[135]:


import csv

with open('saubja_val.csv','w') as f:
    w = csv.writer(f)
    w.writerow(bbox_list.keys())
    w.writerow(bbox_list.values())


# In[161]:


import csv
import re
import ast
bbox_list = {}
with open('saubja_val.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        for i,j in zip(row.keys(), row.values()):
            bbox_list[i] =  ast.literal_eval(j)
            
bbox_list = {'value':bbox_list['value'],'value_label':bbox_list['value_label']}
bbox_list


# # 표가 있는 양식 (CROP)

# In[162]:


regist_img_num = 0
# test_img_num = 1

# load net
net = CRAFT()     # initialize

print('Loading weights from checkpoint (' + args.trained_model + ')')
device=torch.device('cuda')
if args.cuda:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model,map_location=device)))
else:
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

if args.cuda:
#     net = net.cuda()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

net.eval()

# LinkRefiner
refine_net = None
if args.refine:
    from refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()
    args.poly = True

t = time.time()
image_boxes = []
# load data
for k, image_path in enumerate(image_list,0):
    print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(image, dsize=(3000, 4000))
#     canvas_size = max(img.shape[0],img.shape[1])
    
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     height, width, channel = img.shape
#     _,img_bin = cv2.threshold(gray,180,224,cv2.THRESH_BINARY)
#     img_bin =~img_bin
#     line_min_width = 15
#     kernal_h = np.ones((1,line_min_width), np.uint8)
#     kernal_v = np.ones((line_min_width,1), np.uint8)

#     img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
#     img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

#     img_bin_final = img_bin_h | img_bin_v

#     final_kernel = np.ones((3,3), np.int8)
#     img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
#     _,labels,stats,_ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype= cv2.CV_32S)

#     n1 = np.array(stats[2:])
#     # print(n1)
#     xw = n1[:, [0,2]].sum(axis=1).max()
#     # print('xw',xw)
#     yh = n1[:, [1,3]].sum(axis=1).max()
#     # print('yh',yh)
#     min_x = n1[:, 0].min()
#     min_y = n1[:, 1].min()

#     aa = cv2.rectangle(img,(min_x,min_y),(xw,yh),(0,255,0),4)
#     crop = aa[min_y:yh, min_x:xw]

    
    if k == int(regist_img_num):
#         bbox_list['key'].extend(bbox_list['value'])
        pts = [np.array(i).reshape(-1,2) for i in bbox_list['value']]
#         pts.extend([np.array(i).reshape(-1,2).astype(np.int32) for i in bbox_list['value']])
        
        image_boxes.append([img, pts, labels[k]])
    else:
        bboxes, polys, score_text = test_net(net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
#     print(bboxes)
        bboxes = [np.array(i).reshape(-1,2) for i in bboxes]
        image_boxes.append([img, bboxes, labels[k]])
        
t_2 = time.time() - t
print(f"time : {t_1+t_2}s")
print("elapsed time : {}s".format(time.time() - t))


# ### Detection image 확인

# In[84]:


for idx,i in enumerate(image_boxes):
# i=image_boxes[1]
    img_1= i[0].copy()
    #     cv2.imwrite(f'crop_{idx}_toji.jpeg',i[0])
    for pts in i[1]:
    # for pts,val in zip(i[1],bbox_list['value']):
    # #         print(pts)
    # #         print('sss')
        pts = np.array(pts).reshape(-1,2).astype(np.int32)
    #     val = np.array(val).reshape(-1,2).astype(np.int32)
    # #         print(pts)
        img_1 = cv2.polylines(img_1, [pts], True, (0,0,255),2)
#     for i in bbox_list['value']:
#     #     print(i)

#         i = np.array(i).reshape(-1,2).astype(np.int32)
#     #     print(i)
#         img_1 = cv2.polylines(img_1, [i], True, (255,0,0),2)
    img_show(img_1)


# # text from img func

# In[163]:


def cut_image(bboxes, image):
    images = []
    data_list = []
#     if len(bboxes) != 0:
#     print(bboxes)
    for pts in bboxes:
        pts = pts.astype(np.float32)
        data_list.append([pts[0][0], pts[0][1], pts[1][0], pts[2][1]])

        rect = pts
#         print(rect)
        (top_left, top_right, bottom_right, bottom_left) = rect

        w1 = abs(bottom_right[0] - bottom_left[0])
        w2 = abs(top_right[0] - top_left[0])
        h1 = abs(top_right[1] - bottom_right[1])
        h2 = abs(top_left[1] - bottom_left[1])
#         print(w1,w2,h1,h2)
        max_width = max([w1, w2])
        max_height = max([h1, h2])

        dst = np.float32([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]])
#         print(dst)
        m = cv2.getPerspectiveTransform(rect, dst)
        
        warped = cv2.warpPerspective(image, m, (int(max_width), int(max_height)))
        images.append(warped)

    return images, data_list


# ## OCR

# In[164]:


class Args(object):
    def __init__(self):
        pass

    def add_argument(self, key, default=None, help=None,action=None,required=False,type=None):
        key = key.replace('-','')
        if action == 'store_true':
            self.__dict__[key] = False
        else:
            self.__dict__[key] = default
            
        if required:
            print(key,'/',help)
            
            
    def set_argument(self, data_dict):
        for key in data_dict:
            self.__dict__[key] = data_dict[key]
            


# ### 실행

# In[165]:


parser = Args()

parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', default=True, action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')


# #### 모델 경로

# In[166]:


# model_path = '/workspace/DBP/서류ocr인식/saved_models/200928_TPS-ResNet-BiLSTM-Attn-Seed1119/'
# model_path = '/workspace/DBP/서류ocr인식/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1119/'
save_model_name = 'best_accuracy2.pth'
# save_model_name = 'best_norm_ED.pth'


# In[167]:


# # model_path = '/workspace/DBP/서류ocr인식/saved_models/201021_TPS-ResNet-BiLSTM-Attn-Seed1119/'
# model_path = '/workspace/DBP/서류ocr인식/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1119/'
# save_model_name = 'best_accuracy.pth'
# # save_model_name = 'best_norm_ED.pth'


# In[168]:


with open('opt.txt','r') as f:
    opt = f.read()
    
opt = opt.split('------------ Options -------------\n')
opt = opt[-1].split('\n')

# test 실행용
opt_dict ={}
int_keys = ['manualSeed', 'workers', 'batch_size', 'num_iter', 'valInterval', 'batch_max_length', 'imgH', 'imgW', 'num_fiducial', 'input_channel', 'output_channel', 'hidden_size']
float_keys = ['lr', 'beta1', 'rho', 'eps', 'grad_clip']
bool_keys = ['FT', 'adam', 'rgb', 'sensitive', 'PAD', 'data_filtering_off']
str_keys = ['exp_name', 'train_data', 'valid_data', 'saved_model', 'select_data', 'batch_ratio', 'total_data_usage_ratio', 'character', 'Transformation', 'FeatureExtraction', 'SequenceModeling', 'Prediction']
save_keys = ['image_folder', 'workers', 'batch_size', 'saved_model', 'batch_max_length', 'imgH', 'imgW', 'rgb', 'character', 'sensitive', 'PAD', 'Transformation', 'FeatureExtraction', 'SequenceModeling', 'Prediction', 'num_fiducial', 'input_channel', 'output_channel', 'hidden_size','rgb']
for i in opt:
    t = i.split(':')
    if i == '---------------------------------------':
        break
    else:
        key, data = t[0].strip(), t[1][1:]
        if key in int_keys:
            data = int(data)
        elif key in float_keys:
            data = float(data)
        elif key in bool_keys:
            if data == 'True':
                data = True
            elif data == 'False':
                data = False      
        elif key in str_keys:
            data = str(data)
        if key in save_keys:
            opt_dict[key] = data
    


# In[169]:


parser.set_argument(opt_dict)
parser.sensitive = True
parser.character = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝!@#$%^&*《》()[]【】【】\"\'◐◑oㅇ⊙○◎◉◀▶⇒◆■□△★※☎☏;:/.?<>-_=+×\￦|₩~,.㎡㎥ℓ㎖㎘→「」『』·ㆍ1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ읩①②③④⑤月日軍 '
parser.image_folder = image_list
parser.saved_model = save_model_name
cudnn.benchmark = True
cudnn.deterministic = True

opt = parser


# In[170]:


from sklearn.preprocessing import MinMaxScaler

def merging_area(temp,img):
    
    temp['left_min_x_p'] = [np.min([x1, x4]) for x1, x4 in temp[['x1', 'x4']].values.tolist()]
    temp['right_max_x_p'] = [np.min([x2, x3]) for x2, x3 in temp[['x2', 'x3']].values.tolist()]
    temp['top_min_y_p'] = [np.min([y1, y2]) for y1, y2 in temp[['y1', 'y2']].values.tolist()]
    temp['bottom_max_y_p'] = [np.max([y4, y3]) for y4, y3 in temp[['y4', 'y3']].values.tolist()]

    ## 오른쪽변 높이 추출
    temp['right_height'] = temp['y3'] - temp['y2']

    temp = temp.sort_values(['y1', 'x1'])
    temp['unique_no'] = temp.index

    for idx, row in temp.iterrows():
        merge_target = temp[(row['right_max_x_p'] <= temp.left_min_x_p)]
    #     print(row)
        merge_target = merge_target[(row['top_min_y_p'] - row['right_height'] / 2) <= merge_target.top_min_y_p]

        merge_target = merge_target[(row['bottom_max_y_p'] + row['right_height'] / 2) >= merge_target.bottom_max_y_p]
        
        inner_df = merge_target.copy()
        
        merge_target = merge_target[(row['right_max_x_p'] + row['right_height']) >= merge_target.left_min_x_p]
        
#         if len(inner_df.ocr) ==1:
#             merge_target = pd.concat([merge_target,inner_df])
            
        if merge_target.shape[0] != 0:

            temp.loc[merge_target.index, 'unique_no'] = row['unique_no']

            temp = temp.sort_index()

        else:

            temp.loc[idx, 'x2'],temp.loc[idx, 'x3'] = row['x2']+ row['right_height']/2 ,row['x3']+ row['right_height']/2

    ### 병합된 좌표 추출
    p1 = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no')[['x1', 'y1', 'x4', 'y4']].head(1).values
    p2 = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no')[['x2', 'y2', 'x3', 'y3']].tail(1).values
    ocr_df = pd.DataFrame(np.concatenate((p1, p2), axis=1), columns=['x1', 'y1', 'x4', 'y4', 'x2', 'y2', 'x3', 'y3'])

    ### ocr 병합
    ocr_df['ocr'] = temp.sort_values(['unique_no', 'left_min_x_p']).groupby('unique_no').ocr.apply(lambda x: ' '.join(x)).values
    ocr_df = ocr_df[['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'ocr']]        
    ### 정렬

    return ocr_df


# In[171]:


# from predict import ocr_predict
import pandas as pd
import torch
import torch.utils.data
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
import pytesseract
from utill import CTCLabelConverter, AttnLabelConverter
from ocrmodel import Model
from dataset import AlignCollate, RawDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

def ocr_predict(opt, images):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
   
    
#     print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
#           opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
#           opt.SequenceModeling, opt.Prediction)
    
    model = Model(opt)
    
#     if device == "cuda":
#         print(device)
#         model.to(device)
    model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        
#     else:
#         model.load_state_dict(torch.load(opt.saved_model, map_location=device),strict=False)
    # load model
#     print('loading pretrained model from %s' % opt.saved_model)
#     model.to(device)
    
#     model.load_state_dict(torch.load(opt.saved_model, map_location=device)['model_state_dict'])
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=images, opt=opt)  # use RawDataset
    
#     if device=='cuda':
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
#     else:
#         demo_loader = torch.utils.data.DataLoader(
#             demo_data, batch_size=opt.batch_size,
#             shuffle=False,
#             num_workers=int(opt.workers),
#             collate_fn=AlignCollate_demo)
#     print('데이터 생성완료')

    # predict
    model.eval()
    result_list = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
      
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
#             print('데이터 텐서화')

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            #             log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            #             print(f'{dashed_line}\n{head}\n{dashed_line}')
            #             log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            count = 0
            
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                count += 1
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score  = 0
                result_list.append([img_name, pred, float(confidence_score)])
    #                 print(f'{img_name}\t{pred:25s}\t{confidence_score:0.4f}')
    #                 log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
    #                 img_show(img_name)
    #                 print(pred,confidence_score)

    #             log.close()
    return result_list


# In[172]:


def merge_predict(bboxes, result_list):

    # # OCR 결과 Dataframe 생성
    bboxes = np.array(bboxes).astype(int)
    bboxes_df = pd.DataFrame(bboxes.reshape(-1, 8), columns=['x1','y1','x2','y2','x3','y3','x4','y4'])
    temp = np.array(result_list)
    bboxes_df['ocr'] = temp[:, 1]
#     bboxes_df['per'] = temp[:, 2]
    merge_df = bboxes_df
#     merge_df = merging_area(bboxes_df)
    merge_df['height'] = merge_df['y4']-merge_df['y1']
    merge_df['width'] = merge_df['x2']-merge_df['x1']
    return merge_df

def merge_predict2(bboxes, result_list,img):

    # # OCR 결과 Dataframe 생성
    
    bboxes = np.array(bboxes).astype(int)
    bboxes_df = pd.DataFrame(bboxes.reshape(-1, 8), columns=['x1','y1','x2','y2','x3','y3','x4','y4'])
    
    temp = np.array(result_list)
    bboxes_df['ocr'] = temp[:, 1]
#     bboxes_df['per'] = temp[:, 2]
    merge_df = bboxes_df.copy()
    default_df = bboxes_df.copy()
    for i in range(5):
#         print(merge_df)
        merge_df = merging_area(merge_df,img)
#         if i == 0:
#             merge_df = merge_df
        bboxes_df = pd.concat([bboxes_df,merge_df])
#     bboxes_df = pd.concat([bboxes_df,merge_df])

    bboxes_df['height'] = bboxes_df['y4']-bboxes_df['y1']
    bboxes_df['width'] = bboxes_df['x2']-bboxes_df['x1']
    default_df['height'] = default_df['y4']-default_df['y1']
    default_df['width'] = default_df['x2']-default_df['x1']
    
    return bboxes_df,default_df


# In[173]:


def img_erode(img, kernel_shape, iterations=3):
    kernel = np.ones(kernel_shape, np.uint8)
    img = cv2.erode(img, kernel, iterations = 3)
    return img


def set_imgs_margin(imgs, fill_cnt = 40, is_color = False):
    h_fill_cnt = int(fill_cnt/2)
    new_imgs = []
    len(imgs)
    for i in imgs:
        if is_color:
            new_bg_img = np.zeros([i.shape[0]+fill_cnt,i.shape[1]+fill_cnt,3],dtype='uint8')
            new_bg_img.fill(255)
            new_bg_img[h_fill_cnt:(h_fill_cnt*-1),h_fill_cnt:(h_fill_cnt*-1)] = i
            new_imgs.append(new_bg_img)
        else:
            new_bg_img = np.zeros([i.shape[0]+fill_cnt,i.shape[1]+fill_cnt],dtype='uint8')
            new_bg_img.fill(255)
            new_bg_img[h_fill_cnt:(h_fill_cnt*-1),h_fill_cnt:(h_fill_cnt*-1)] = i
            new_imgs.append(new_bg_img)
    return new_imgs


def get_ocr_data(img, output_type='string', is_gray_scaler= True, debug=True, is_color=False, processing_type = 1):
    #tessdata_dir = '--tessdata-dir ' + '/usr/share/tesseract-ocr/4.00/tessdata --psm 6'
#     tessdata_dir = '--tessdata-dir /Users/soncheoljun/ocr_api/ocsapi/tessdata/40 --psm 6'
    tessdata_dir = '--tessdata-dir ' + '/workspace/DBP/서류ocr인식/tessdata --psm 1' 
    ori_img = img.copy() 
    
    if is_gray_scaler:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = gray_scale(img)
    
    if processing_type == 1:
        size = 100
        r = size/img.shape[0]
        dim = (int(img.shape[1]*r),size)
        if dim[0] !=0 and dim[1] !=0:
            img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
        img = img_erode(img,2)
    elif processing_type == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        border = cv2.copyMakeBorder(img,2,2,2,2,   cv2.BORDER_CONSTANT,value=[255,255])
        resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dilation = cv2.dilate(resizing, kernel,iterations=1)
        erosion = cv2.erode(dilation, kernel,iterations=1)
            
    if is_color == False:
        img = set_imgs_margin([img],is_color=False)[0]

    if output_type == 'dataframe':
        d = pytesseract.image_to_data(img, lang='kor', output_type=Output.DICT, config=tessdata_dir)
        return_data = pd.DataFrame.from_dict(d)
        string = return_data['text'][0]
    elif output_type == 'string':
        string = pytesseract.image_to_string(img, lang='kor', config=tessdata_dir)
        return_data = string
    elif output_type == 'list':
        d = pytesseract.image_to_data(img, lang='kor', output_type='data.frame', config=tessdata_dir)
        d = d[d.conf != -1]
        
        lines = d.groupby('block_num')['text'].apply(lambda x: ''.join(x)).tolist()
        conf = d.groupby(['block_num'])['conf'].mean().tolist()
        return_data = [lines,conf]
        
    if debug:
        img_show(img)
        print(string)
    
    return return_data


# In[174]:


#image_boxes: image, bboxes, polys, score_text
kernel = np.ones((1, 1), np.uint8)
results = []
t = time.time()
for idx,img in enumerate(image_boxes,0):
    if idx != regist_img_num:
    # img =  image_boxes[test_img_num]
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        if len(img[1]) != 0:

            images, _ = cut_image(img[1], gray)

            parser.image_folder = images
            opt = parser

            bboxes = img[1]
            result_list = ocr_predict(opt, images)
            
            for r_idx in result_list:

                if r_idx[-1] < 0.7:
                    img_t = cv2.dilate(images[r_idx[0]], kernel, iterations=1)
                    img_t = cv2.erode(img_t, kernel, iterations=1)
                    img_t = cv2.threshold(cv2.medianBlur(img_t, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                    d = pytesseract.image_to_data(img_t, lang='kor', output_type='data.frame', config='--psm 6')

                    d = d[d.conf != -1]

                    lines = d.groupby('block_num')['text'].apply(list)
                    conf = d.groupby(['block_num'])['conf'].mean().tolist()

                    if len(conf) != 0:

                        if conf[0] >= 70:

                            lines = [str(i) for i in lines.item()]

                            ocr_str = ''.join(lines)

                            result_list[r_idx[0]][1] = ocr_str

#             print(result_list)
            result_list, def_df = merge_predict2(bboxes, result_list,img[0])
            results.append([result_list,def_df,img[0]])

t_3 = time.time() - t
print(f"time : {t_1+t_2+t_3}s")


# In[175]:


# -*- coding: utf-8 -*-
import re
import sys
import numpy as np

# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

def convert(test_keyword):
    split_keyword_list = list(test_keyword)
    
    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            
            char_code = ord(keyword) - BASE_CODE

            if char_code >= 0: 
                char1 = int(char_code / CHOSUNG)

                result.append(CHOSUNG_LIST[char1])

    #             print('result1',result)
                char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
                result.append(JUNGSUNG_LIST[char2])
    #             print('result2',result)
                char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
                if char3==0:
                    continue
    #                 result.append('#')
                else:
                    result.append(JONGSUNG_LIST[char3])
#             print('result3',result)
   
        else:
            result.append(keyword)
    return result


def char_error_cnt(target_text, text, return_type='cnt'):
#     text = text.replace(' ', '')
    
    docompose_target_text = convert(target_text)
    s_decompose_text = convert(text)
    
    t_size = len(docompose_target_text)
    len_diff = np.abs(t_size - len(s_decompose_text))
    cer_error_cnt_list = []
    if (t_size - len(s_decompose_text)) > 0:
        
        for i in range(len_diff+1):
            
            sample_text = s_decompose_text
            sample_text += ['']*(len(docompose_target_text) - len(s_decompose_text)-i) # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]
#             print(sample_text)
            if i >0:
                sample_text.insert(0,'')
                sample_text.pop(-1)
#             print(sample_text,docompose_target_text)
            cer_error_cnt = (np.array(sample_text) == np.array(docompose_target_text)).sum()
            cer_error_cnt_list.append(cer_error_cnt)
            
            
        
    elif (t_size - len(s_decompose_text)) == 0:
        
        sample_text = s_decompose_text

        cer_error_cnt = (np.array(sample_text) == np.array(docompose_target_text)).sum()
        cer_error_cnt_list.append(cer_error_cnt)
            
    else:
        
        for i in range(len_diff+1):
            sample_text = docompose_target_text
            sample_text += ['']*(len(s_decompose_text)-len(docompose_target_text)-i) # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]

            if i >0:
                sample_text.insert(0,'')
                sample_text.pop(-1)
            
            cer_error_cnt = (np.array(sample_text) == np.array(s_decompose_text)).sum()
            cer_error_cnt_list.append(cer_error_cnt)
            
    if return_type == 'cnt':
        return np.max(cer_error_cnt_list)
    
    if return_type == 'per':
        if (t_size - len(s_decompose_text)) < 0:
            return np.max(cer_error_cnt_list) / len(s_decompose_text)
        else:
            return np.max(cer_error_cnt_list) / t_size
        
def char_error_cnt2(target_text, text, return_type='cnt'):
#     text = text.replace(' ', '')
    
    docompose_target_text = convert(target_text)
    s_decompose_text = convert(text)
    
    t_size = len(docompose_target_text)
    len_diff = np.abs(t_size - len(s_decompose_text))
    cer_error_cnt_list = []
            
    for i in range(len_diff+1):
        sample_text = docompose_target_text
        sample_text += ['']*(len(s_decompose_text)-len(docompose_target_text)-i) # for i in range(len(docompose_target_text) - len(s_decompose_text)-c)]

        if i > 0:
            sample_text.insert(0,'')
            sample_text.pop(-1)

        cer_error_cnt = (np.array(sample_text) == np.array(s_decompose_text)).sum()
        cer_error_cnt_list.append(cer_error_cnt)
            
    if return_type == 'cnt':
        return np.max(cer_error_cnt_list)
    
    if return_type == 'per':
            
        return np.max(cer_error_cnt_list) / t_size        


# In[176]:


def IoU(box1, box2): # box2가 regist (기준)
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection

    # w,h = 0,0

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    
    inter = w * h
#     iou = inter / (box1_area + box2_area - inter)
    iou = inter/ box1_area
#     if inter > 0 and inter <= box1_area and inter/box2_area >=0.8:
#         iou = 1
        

    return iou


# In[152]:


import re
t = time.time()
docu_name = ['사업자등록증','토지대장','특허증']
doc_dic={}
check_docu = ''.join([q[-3] for q in results[0][0].values.tolist()]).replace(" ","")

check_docu = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', check_docu).replace(" ","")
for docu in docu_name:
    if char_error_cnt2(docu, check_docu, return_type='per') >= 0.8:
        docu_name = docu
        
        t_docu = time.time() -t
        print(f'time: {t_docu}')
        print(f'이 문서는 {docu} 입니다' + '\n')


with open('saubja_name.csv', 'rU') as p:
    
    for idx,ii in enumerate(csv.reader(p,delimiter=',')):

        for i in range(0,len(ii)):

            ii[i] = ii[i].replace("[",'').replace("]",'')
            ii[i] = re.split('[,]',ii[i])
        for k in range(len(ii)):
            for l in range(len(ii[k])):
    #                 print(ii[k][l])
                ii[k][l] = float(ii[k][l])

        doc_dic['name'] = docu_name
        doc_dic['xy'] = ii
doc_dic


# In[177]:


# from char_error import char_error_cnt


t = time.time()

# regist_df = results[regist_img_num][0].copy()
for res in range(len(results)):
        
    test_img_cp = results[res][-1].copy()
    regist_x = [i[:,0]/image_boxes[regist_img_num][0].shape[1]*results[res][-1].shape[1] for i in image_boxes[regist_img_num][1]]
    regist_y = [i[:,1]/image_boxes[regist_img_num][0].shape[1]*results[res][-1].shape[1] for i in image_boxes[regist_img_num][1]]
    test_df = results[res][0].copy()
    test_df[['x1','x2','x3','x4']] = test_df[['x1','x2','x3','x4']]/results[res][-1].shape[1]
    test_df[['y1','y2','y3','y4']] = test_df[['y1','y2','y3','y4']]/results[res][-1].shape[0]
    
    transform = {'label':[],'detection':[]}
    
#     for idx1, i in enumerate(bbox_list['value']):
    
    regist_1st_x = doc_dic['xy'][0][0]/image_boxes[regist_img_num][0].shape[1]
    regist_1st_y = doc_dic['xy'][0][1]/image_boxes[regist_img_num][0].shape[0]

    text = doc_dic['name']
    text_split = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text).replace(" ","")
    diff_x, diff_y = 0,0
    for idx2, k in enumerate(test_df.values.tolist()):

        text2 = ''.join([q for q in k[-3]]).replace(" ","")
        text2 = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text2).replace(" ","")
#         print(f'text:{text_split} text2: {text2}')
        if char_error_cnt(docu_name, text2, return_type='per') > 0.9 and char_error_cnt(docu_name, text_split, return_type='per') > 0.9:
#             print(f'text:{text_split} text2: {text2}')
            test_img = k[:8]

            test_1st_x,test_1st_y = test_img[0] , test_img[1]

            diff_x, diff_y = (test_1st_x-regist_1st_x)*results[res][-1].shape[1], (test_1st_y-regist_1st_y)*results[res][-1].shape[0]

#                 regist_xy = regist_df.iloc[:,0:8].values.tolist()

    trans_x = [[t[0]+ diff_x, t[1]+ diff_x, t[2]+ diff_x,t[3]+ diff_x] for t in regist_x]
    trans_y = [[g[0]+ diff_y, g[1]+ diff_y, g[2]+ diff_y,g[3]+ diff_y] for g in regist_y]

    for idx,(x, y) in enumerate(zip(trans_x, trans_y)):

        pts1 = np.array([x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]]).reshape(-1,2).astype(np.int32)

        transform['label'].append(bbox_list['value_label'][idx])
        transform['detection'].append([x[0],y[0],x[1],y[1],x[2],y[2],x[3],y[3]])

        test_img_cp = cv2.polylines(test_img_cp,[pts1],True,(0,255,0),2)

    result_df = pd.DataFrame(columns=['KEY','VALUE'])
    
    for idx1,i in enumerate(transform['detection']):
        same_label = []
        box2 = [i[0],i[1],i[4],i[5]]
        for idx2, k in enumerate(results[res][1].iloc[:,0:8].values.tolist()):
            box1 = [k[0],k[1],k[4],k[5]]
            iou_result= IoU(box1,box2)
            if iou_result >= 0.7:
                pts2 = np.array(k[0:8]).reshape(-1,2).astype(np.int32)
                test_img_cp = cv2.polylines(test_img_cp,[pts2],True,(0,0,255),2)
                
                same_label.append([results[res][1].values.tolist()[idx2][-3].split(' ')[0],pts2[0][0],pts2[3][0],pts2[0][1],pts2[3][1]])
        
        same_label = pd.DataFrame(same_label, columns=['text','x1','x4','y1','y4'])
        same_df = pd.DataFrame(columns=['text','x1','x4','y1','y4'])
        if same_label.empty != True:
            for idx,row in enumerate(same_label.iterrows()):
                min_idx = same_label[same_label['y1'] == min(same_label['y1'])]

                min_y4 = min_idx[min_idx['x1'] == min(min_idx['x1'])]['y4'].item()
                same_line = same_label[same_label['y1'] < min_y4]
                same_line = same_line.sort_values(by=['x1'], ascending=True)

                same_df = pd.concat([same_df,same_line])
                same_df = same_df.drop_duplicates(['text','x1','x4','y1','y4'])
                
                same_label = same_label.drop(same_label.index[0])
                

#         print(same_df)
        result_df = result_df.append({'KEY': transform['label'][idx1],'VALUE':' '.join([x[0] for x in same_df.values.tolist()])},ignore_index=True)

    t_4 = time.time() - t
    print(f"time : {t_1+t_2+t_3+t_4}s")
    print(result_df)
    img_show(test_img_cp)            
    


# In[ ]:




