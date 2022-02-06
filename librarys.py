import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import requests
import json
import psycopg2
import json
from psycopg2.extras import execute_values
import random
import time
import os
import string
from PIL import Image

from rest_framework.response import Response
from django.db import connection
from django.conf import settings

from ocr.librarys import *
from craft import imgproc
from doc_classifier.querys import *


# class Connection:
#     def cursor(self,conn_config):
#         # conn = psycopg2.connect(host='10.70.172.75', dbname='personinfo', user='service_app', password='DBP!app123$')
#         # cur = conn.cursor()
#         return conn

def Connection():
    default = settings.DATABASES['default']
    conn = psycopg2.connect(
            dbname=default['NAME'],
            user=default['USER'],
            host=default['HOST'],
            password=default['PASSWORD'],
            port=default['PORT']
            # options=f'-c search_path={schema}',
        )

    return conn

def name_to_json(cursor):
    """
    cursor.fetchall() 함수로 받아온 쿼리 결과를 json 형식으로 만들어 반환해주는 함수입니다.
    :param cursor: SQL 연결 변수
    :return: JSON 쿼리 결과 LIST
    """
    row = [dict((cursor.description[i][0], value)
                for i, value in enumerate(row)) for row in cursor.fetchall()]
    return row


def img_show(img, size =(15,15)):
    plt.rcParams["figure.figsize"] = size
    imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

class Args():
    def __init__(self,cuda=False, trained_model='craft/weights/craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size =1280, mag_ratio=1.5, poly=False, show_time=False,test_folder='/data/',refine=False, refiner_model='weights/craft_refiner_CTW1500.pth'):
        self.cuda = cuda
        self.trained_model = trained_model
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


class Args2(object):
    def __init__(self):
        pass

    def add_argument(self, key, default=None, help=None, action=None, required=False, type=None):
        key = key.replace('-', '')
        if action == 'store_true':
            self.__dict__[key] = False
        else:
            self.__dict__[key] = default

        if required:
            print(key, '/', help)

    def set_argument(self, data_dict):
        for key in data_dict:
            self.__dict__[key] = data_dict[key]


def IoU(box1, box2):  # box2가 regist (기준)
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
    iou = inter / box1_area
    #     if inter > 0 and inter <= box1_area and inter/box2_area >=0.8:
    #         iou = 1

    return iou

def post_request(img_path,end_point='text_detection', add_data={}):
    try:
        myfile = {
            'image': open(img_path, 'rb')
        }

        response = requests.request("POST", verify=False, url=settings.MODEL_API_URL+end_point+'/', files=myfile, data=add_data)

        data = response.text
        if end_point == 'text_detection':
            return True, json.loads(data)['bboxes']
        else:
            return True, json.loads(data)

    except Exception as e:
        print('detection_error', e)
        return False, e


class api_model():
    def __init__(self,endpoint,bbox_info=[],regist_sh=[],test_sh=[],resize_sh=[]):
        self.endpoint = endpoint
        self.regist_sh = regist_sh
        self.test_sh = test_sh
        self.bbox_info = bbox_info
        self.resize_sh = resize_sh

    def get_img(self,doc_path,company_no):

        n = 20  # 문자의 개수(문자열의 크기)
        rand_str = ""  # 문자열
        for i in range(n):
            rand_str += str(random.choice(string.ascii_lowercase))

        extension = str(doc_path).split('.')[-1]
        save_name = str(company_no) + '_' + str(rand_str) + '.' + extension

        with open(settings.SAVE_IMG_PATH + save_name, 'wb') as w:

            for chunk in doc_path.chunks():
                w.write(chunk)

        local_name = settings.MEDIA_URL + '?img_name=' + save_name

        return save_name,local_name

    def load_img(self,save_name):

        img_path = os.path.join(settings.MEDIA_PATH, save_name)
        img = imgproc.loadImage(img_path)

        return img_path,img

    def ocr_to_merge(self,ocr_data,img):
        result_list = {'bboxes': [], 'ocr': []}

        for i in ocr_data.get('data', []):
            # if float(i['per']) >= 0.5:
            pts = i['vertices']
            result_list['bboxes'].append([pts['x1'], pts['y1'], pts['x2'], pts['y2'], pts['x3'], pts['y3'], pts['x4'], pts['y4']])
            result_list['ocr'].append(i['ocr'])
            # pts = np.array([pts['x1'], pts['y1'], pts['x2'], pts['y2'], pts['x3'], pts['y3'], pts['x4'], pts['y4']]).reshape(-1, 2).astype(np.int32)  # 텍스트 추출 테스트
            # test_img_cp = cv2.polylines(test_img_cp, [pts], True, (255, 255, 0), 5)  # 텍스트 추출 테스트
        t = time.time()

        result_list, def_df = merge_predict2(result_list['bboxes'], result_list['ocr'], img)

        print('merging time:', time.time() - t)
        results = [result_list, def_df, img]

        return results

    def template_key_far(self):

        # info_list = []
        # if self.endpoint == 'classify':
        #     info_list = [[sr_k,k] for sr_k, sr_v,k in zip(self.bbox_info['sr_keyword'],self.bbox_info['sr_value'],self.bbox_info['keyword']) if sr_v['p1'] !=[0,0] and sr_v['p3'] !=[0,0] and len(sr_k) != 0]
        #     info_list.insert(0, [self.bbox_info['sr_value'][0],self.bbox_info['keyword'][0]])
        #     info_list.sort(key=lambda x: x[0]['p4'][1])


        # elif self.endpoint == 'document_form':

        # info_list = [[info['sr_keyword'],info['keyword']] for info in self.bbox_info if info['sr_value']['p1'] !=[0,0] and info['sr_value']['p3'] !=[0,0]]
        form_info = [[info['sr_keyword'], info['sr_value'], info['keyword']] for info in self.bbox_info if
                          info['sr_value']['p1'] != [0, 0] and info['sr_value']['p3'] != [0, 0]]

        form_info.sort(key=lambda x: x[0]['p4'][1])

        info_list = form_info.copy()
        info_list[0] = [self.bbox_info[0]['sr_value'], self.bbox_info[0]['sr_keyword'], self.bbox_info[0]['keyword']]
        info_list.sort(key=lambda x: x[0]['p4'][1])

        idx = 0
        template_far = []
        while idx < len(info_list):
            next_info, info = info_list[idx + 1][0], info_list[idx][0]
            if int(idx) != int(len(info_list)-1):
                template_far.append([(next_info['p1'][0] + ((next_info['p2'][0] - next_info['p1'][0]) / 2) - info['p1'][0] + ((info['p2'][0] - info['p1'][0]) / 2))/self.regist_sh[1],
                                     (next_info['p1'][1] + ((next_info['p4'][1] - next_info['p1'][1]) / 2) - info['p1'][1] + ((info['p4'][1] - info['p1'][1]) / 2))/self.regist_sh[0],
                                      info_list[idx][2]])
            else:
                template_far.append([0,0,info_list[idx][2]])

            idx += 1

        return template_far, len(info_list)


    def find_samekey(self,results,key_count):
        start = 0
        arr_ocr = []

        bbox_info_cp = [
            [info['sr_value'], info['sr_keyword'], info['keyword']] if info['sr_value']['p1'] != [0, 0] and info['sr_value']['p3'] != [0, 0] else [info['sr_keyword'], info['sr_value'], info['keyword']] for idx,info in enumerate(self.bbox_info)].copy()

        for idx1 in range(start, len(bbox_info_cp)):

            info = bbox_info_cp[idx1]

            if info[1]['p1'] != [0, 0] and info[1]['p3'] != [0, 0]:
                acc_list = []
                diff_cen_list = []
                append_list = []
                text = info[2]

                text_split = convert_to_text(text)

                for idx2, k in enumerate(results[0].values.tolist()):

                    text2 = ''.join([q for q in k[-3]])

                    text2 = convert_to_text(text2)

                    acc_cer2 = char_error_cnt(text_split, text2, return_type='per')

                    if acc_cer2 > 0.8:

                        test_kbox = k[0:8]

                        test_list = [
                            (k[0] + (k[2] - k[0]) / 2) / self.test_sh[1],
                            (k[1] + (k[7] - k[1]) / 2) / self.test_sh[0]]

                        if idx1 == 0:
                            key_box = bbox_info_cp[idx1][0]

                            resz_cen_x = (key_box['p1'][0] + (key_box['p2'][0] - key_box['p1'][0]) / 2) / self.regist_sh[1]
                            resz_cen_y = (key_box['p1'][1] + (key_box['p4'][1] - key_box['p1'][1]) / 2) / self.regist_sh[0]

                        else:
                            acc_list.append(acc_cer2)
                            append_list.append([[text, self.bbox_info[idx1]['sr_keyword'], self.bbox_info[idx1]['sr_value']], [text2, test_kbox]])

                            key_box = bbox_info_cp[idx1][1]

                            resz_cen_x = (key_box['p1'][0] + (key_box['p2'][0] - key_box['p1'][0]) / 2) / self.regist_sh[1]
                            resz_cen_y = (key_box['p1'][1] + (key_box['p4'][1] - key_box['p1'][1]) / 2) / self.regist_sh[0]

                        diff_cen_xy = [(test_list[0] - resz_cen_x) * self.regist_sh[1], (test_list[1] - resz_cen_y) * self.regist_sh[0]]
                        diff_cen_list.append(diff_cen_xy)

                diff_x, diff_y = 0, 0
                if len(acc_list) > 0 and len(append_list) > 0:
                    best_acc = np.argmax(acc_list)

                    diff_x,diff_y = diff_cen_list[best_acc]
                    arr_ocr.append(append_list[best_acc])
                    # if len(append_list) > 0:
                    #
                    #     arr_ocr.append(append_list[best_acc])

                if key_count > 4:
                    for ids, (cr_val, cr_key, keyword) in enumerate(bbox_info_cp):
                        if idx1 == 0:
                            bbox_info_cp[ids][0] = {'p1': [cr_val['p1'][0] + diff_x,
                                                           cr_val['p1'][1] + diff_y],
                                                    'p2': [cr_val['p2'][0] + diff_x,
                                                           cr_val['p2'][1] + diff_y],
                                                    'p3': [cr_val['p3'][0] + diff_x,
                                                           cr_val['p3'][1] + diff_y],
                                                    'p4': [cr_val['p4'][0] + diff_x,
                                                           cr_val['p4'][1] + diff_y]}

                            bbox_info_cp[ids][1] = {'p1': [cr_key['p1'][0] + diff_x,
                                                           cr_key['p1'][1] + diff_y],
                                                    'p2': [cr_key['p2'][0] + diff_x,
                                                           cr_key['p2'][1] + diff_y],
                                                    'p3': [cr_key['p3'][0] + diff_x,
                                                           cr_key['p3'][1] + diff_y],
                                                    'p4': [cr_key['p4'][0] + diff_x,
                                                           cr_key['p4'][1] + diff_y]}

                        else:
                            idx_keyword = list(map(lambda x: x[2], bbox_info_cp)).index(str(text))

                            if bbox_info_cp[idx_keyword][1]['p1'][1] < cr_val['p1'][1]:
                                bbox_info_cp[ids][0] = {'p1': [cr_val['p1'][0] + diff_x,
                                                               cr_val['p1'][1] + diff_y],
                                                        'p2': [cr_val['p2'][0] + diff_x,
                                                               cr_val['p2'][1] + diff_y],
                                                        'p3': [cr_val['p3'][0] + diff_x,
                                                               cr_val['p3'][1] + diff_y],
                                                        'p4': [cr_val['p4'][0] + diff_x,
                                                               cr_val['p4'][1] + diff_y]}

                else:
                    if idx1 == 0:
                        for ids, (cr_val, cr_key, keyword) in enumerate(bbox_info_cp):
                            bbox_info_cp[ids][0] = {'p1': [cr_val['p1'][0] + diff_x,
                                                           cr_val['p1'][1] + diff_y],
                                                    'p2': [cr_val['p2'][0] + diff_x,
                                                           cr_val['p2'][1] + diff_y],
                                                    'p3': [cr_val['p3'][0] + diff_x,
                                                           cr_val['p3'][1] + diff_y],
                                                    'p4': [cr_val['p4'][0] + diff_x,
                                                           cr_val['p4'][1] + diff_y]}


        test_xy = results[1].iloc[:, 0:9]

        trans = []
        for cr_val, cr_key, keyword in bbox_info_cp:
            trans.append([[cr_val['p1'][0] * self.resize_sh[1], cr_val['p1'][1] * self.resize_sh[0],
                           cr_val['p2'][0] * self.resize_sh[1], cr_val['p2'][1] * self.resize_sh[0],
                           cr_val['p3'][0] * self.resize_sh[1], cr_val['p3'][1] * self.resize_sh[0],
                           cr_val['p4'][0] * self.resize_sh[1], cr_val['p4'][1] * self.resize_sh[0]],
                          keyword])

        return arr_ocr, test_xy, trans


    def model(self,test_img_cp,results,num_test=0):

        # template_far, key_count = api_model.template_key_far(self)
        form_info = [[info['sr_keyword'], info['sr_value'], info['keyword']] for info in self.bbox_info if info['sr_value']['p1'] != [0, 0] and info['sr_value']['p3'] != [0, 0]]
        key_count = len(form_info)

        arr_ocr, test_xy, trans = api_model.find_samekey(self, results, key_count)

        value_dic = {'label': [], 'detection': [], 'value_text': []}
        point_rec_box_list = []
        for regist, test in arr_ocr:

            regist_h = (regist[1]['p4'][1] - regist[1]['p1'][1]) / self.regist_sh[0]

            point_rec_box = [
                [(test[1][0] / self.test_sh[1] + (regist[2]['p1'][0] - regist[1]['p1'][0]) / self.regist_sh[1]) * self.test_sh[1] - (regist_h / 2 * self.test_sh[0]),
                 (test[1][1] / self.test_sh[0] + (regist[2]['p1'][1] - regist[1]['p1'][1]) / self.regist_sh[0] - regist_h / 4) * self.test_sh[0]],

                [(test[1][2] / self.test_sh[1] + (regist[2]['p2'][0] - regist[1]['p2'][0]) / self.regist_sh[1]) * self.test_sh[1] + (regist_h / 2 * self.test_sh[0]),
                 (test[1][3] / self.test_sh[0] + (regist[2]['p2'][1] - regist[1]['p2'][1]) / self.regist_sh[0] - regist_h / 4) * self.test_sh[0]],

                [(test[1][4] / self.test_sh[1] + (regist[2]['p3'][0] - regist[1]['p3'][0]) / self.regist_sh[1]) * self.test_sh[1] + (regist_h / 2 * self.test_sh[0]),
                 (test[1][5] / self.test_sh[0] + (regist[2]['p3'][1] - regist[1]['p3'][1]) / self.regist_sh[0] + regist_h / 4) * self.test_sh[0]],

                [(test[1][6] / self.test_sh[1] + (regist[2]['p4'][0] - regist[1]['p4'][0]) / self.regist_sh[1]) * self.test_sh[1] - (regist_h / 2 * self.test_sh[0]),
                 (test[1][7] / self.test_sh[0] + (regist[2]['p4'][1] - regist[1]['p4'][1]) / self.regist_sh[0] + regist_h / 4) * self.test_sh[0]]
            ]
            point_rec_box_list.append(sum(point_rec_box, []))

            box1 = test_xy[['x1', 'y1', 'x3', 'y3']]

            box2 = [point_rec_box[0][0], point_rec_box[0][1], point_rec_box[2][0], point_rec_box[2][1]]

            for ids, test_box in enumerate(box1.values.tolist()):
                if IoU(test_box, box2) >= 0.8:
                    xy = test_xy.values[ids].tolist()[0:8]
                    if xy not in value_dic['detection']:
                        value_dic['label'].append(regist[0])
                        value_dic['detection'].append(xy)
                        value_dic['value_text'].append(test_xy['ocr'].values[ids])

        result_df = []
        # key_info = []
        # if self.endpoint == 'classify':
        #     key_info = self.bbox_info['keyword']
        # elif self.endpoint == 'document_form':
        # key_info = self.bbox_info

        for ids, info in enumerate(self.bbox_info):

            # if self.endpoint == 'document_form':
            keytext = info['keyword']

            same_label,pts_df = [],[]
            for idx, lab in enumerate(value_dic['label']):
                if keytext == lab:

                    pts = np.array(value_dic['detection'][idx]).reshape(-1, 2).astype(np.int32)

                    if len(pts) > 0:
                        if self.endpoint == 'classify':
                            same_label.append([value_dic['value_text'][idx].split(' ')[0], pts[0][0], pts[3][0], pts[0][1], pts[3][1]])
                            pts_df.append([pts[0][0], pts[0][1], pts[2][0], pts[2][1]])

                        elif self.endpoint == 'document_form':
                            same_label.append(
                                [value_dic['value_text'][idx].split(' ')[0], pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1]])

            pts_df = pd.DataFrame(pts_df, columns=['x1', 'y1', 'x3', 'y3'])

            minmax_xy = [pts_df['x1'].min(), pts_df['y1'].min(), pts_df['x3'].max(), pts_df['y3'].max()]

            pts_xy = [minmax_xy[0], minmax_xy[1], minmax_xy[2], minmax_xy[1], minmax_xy[2], minmax_xy[3],minmax_xy[0], minmax_xy[3]]
            # pts_xy = np.array(pts_xy).reshape(-1, 2).astype(np.int32)
            # test_img_cp = cv2.polylines(test_img_cp, [pts_xy], True, (255, 0, 0), 2)
            if self.endpoint == 'classify':
                pts_xy = np.array(pts_xy).reshape(-1, 2).astype(np.int32)

                test_img_cp = cv2.polylines(test_img_cp, [pts_xy], True, (255, 0, 0), 2)
                same_label = pd.DataFrame(same_label, columns=['text', 'x1', 'x4', 'y1', 'y4'])
                same_df = pd.DataFrame(columns=['text', 'x1', 'x4', 'y1', 'y4'])
                same_df = same_label_merge(same_label, same_df)
                result_df.append([int(num_test), ids, keytext, ' '.join([txt for txt in same_df['text'].values]), minmax_xy])

            elif self.endpoint == 'document_form':
                same_label = pd.DataFrame(same_label, columns=['text', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
                same_df = pd.DataFrame(columns=['text', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
                same_df = same_label_merge(same_label, same_df)
                result_df.append([keytext,
                                  ' '.join([txt for txt in same_df['text'].values]),
                                  [{'p1': x[1:3], 'p2': x[3:5], 'p3': x[5:7], 'p4': x[7:9]} for x in same_df.values.tolist()]])

        # transform = {'label': [], 'detection': []}
        for idx, (box,keyword) in enumerate(trans):
            # transform = {}
            # if self.endpoint == 'classify':
            #     transform['label'].append(self.bbox_info['keyword'][idx])
            # elif self.endpoint == 'document_form':
            # transform['label'].append(keyword)
            #
            # transform['detection'].append(box)

        # for idx1, i in enumerate(transform['detection']):
            pts_df,same_label = [],[]
            if idx > 0 :
                box2 = [box[0], box[1], box[4], box[5]]
                # ####drawing test#@@@@
                # ptss = np.array(i).reshape(-1, 2).astype(np.int32)
                # test_img_cp = cv2.polylines(test_img_cp, [ptss], True, (0, 0, 255), 5)
                # ######################
                for idx2, k in enumerate(results[1].iloc[:, 0:8].values.tolist()):

                    box1 = [k[0], k[1], k[4], k[5]]

                    iou_result = IoU(box1, box2)
                    if iou_result >= 0.7:
                        same_label.append([results[1].iloc[idx2,-3].split(' ')[0], k[0], k[4], k[1], k[5]])
                        pts_df.append([k[0], k[1], k[4], k[5]])

            same_label = pd.DataFrame(same_label, columns=['text', 'x1', 'x4', 'y1', 'y4'])
            same_df = pd.DataFrame(columns=['text', 'x1', 'x4', 'y1', 'y4'])
            same_df = same_label_merge(same_label, same_df)
            r_df = [keyword, ' '.join(same_df['text'])]

            if self.endpoint == 'classify':

                if len(result_df[idx][3]) == 0 and len(r_df[1]) > 0:

                    result_df[idx][3] = r_df[1]
                    pts_df = pd.DataFrame(pts_df, columns=['x1', 'y1', 'x3', 'y3'])
                    minmax_xy = [pts_df['x1'].min(), pts_df['y1'].min(), pts_df['x3'].max(), pts_df['y3'].max()]
                    pts_xy = [minmax_xy[0], minmax_xy[1], minmax_xy[2], minmax_xy[1], minmax_xy[2], minmax_xy[3], minmax_xy[0], minmax_xy[3]]
                    iou1 = [pts_xy[0], pts_xy[1], pts_xy[4], pts_xy[5]]

                    for rs in result_df:

                        if IoU(rs[-1], iou1) < 0.7 and IoU(iou1, rs[-1]) < 0.7:

                            pts = np.array(pts_xy).reshape(-1, 2).astype(np.int32)
                            test_img_cp = cv2.polylines(test_img_cp, [pts], True, (255, 0, 0), 2)

            elif self.endpoint == 'document_form':
                if len(result_df[idx][1]) == 0 and len(r_df[1]) > 0:
                    result_df[idx][1] = r_df[1]

        if self.endpoint == 'classify':
            del result_df[0]
            for ids, rs in enumerate(result_df):

                del result_df[ids][-1]

        return result_df,test_img_cp


    def save_draw_image(self,test_img_cp,save_file_name):
        local_img = Image.fromarray(cv2.cvtColor(test_img_cp, cv2.COLOR_BGR2RGB))
        local_img.save(settings.MEDIA_PATH + format(save_file_name))


    def transform(self,template_far, test_kbox_list, key_count, diff_cen_xy):

        if key_count > 4:
            test_far = []
            idx = 0
            while idx < len(test_kbox_list):
                next_info, info = test_kbox_list[idx + 1][0], test_kbox_list[idx][0]
                if int(idx) != int(len(test_kbox_list)-1):
                    test_far.append([(next_info[0] + ((next_info[2] - next_info[0]) / 2) - info[0] + ((info[2] - info[0]) / 2)) / self.test_sh[1],
                                     (next_info[1] + ((next_info[7] - next_info[1]) / 2) - info[1] + ((info[7] - info[1]) / 2)) / self.test_sh[0],
                                     test_kbox_list[idx][1]])
                else:
                    test_far.append([0,0,test_kbox_list[idx][1]])
                idx += 1

            # if self.endpoint == 'document_form':
            bbox_info_cp = [
                [info['sr_value'], info['sr_keyword'], info['keyword']] if info['sr_value']['p1'] != [0, 0] and info['sr_value']['p3'] != [0, 0] else [info['sr_keyword'], info['sr_value'], info['keyword']] for info in self.bbox_info].copy()
            # elif self.endpoint == 'classify':
            #     self.bbox_info = [
            #         [cr_val, cr_key, keyword] if cr_val['p1'] != [0, 0] and cr_val['p4'] != [0, 0] else [cr_key, cr_val, keyword] for cr_val, cr_key, keyword in self.bbox_info]

            for temp_x, temp_y, text1 in template_far:

                if text1 in list(map(lambda x: x[2], test_far)):

                    test_far_idx = list(map(lambda x: x[2],test_far)).index(str(text1))

                    idx_temp = list(map(lambda x: x[2], bbox_info_cp)).index(str(text1))

                    diff_x, diff_y = (test_far[test_far_idx][0] - temp_x) * self.regist_sh[1], (test_far[test_far_idx][1] - temp_y) * self.regist_sh[0]

                    for idx, (cr_val,cr_key,keyword) in enumerate(bbox_info_cp):
                        idx_diff_cen_x, idx_diff_cen_y = diff_cen_xy[idx_temp][0], diff_cen_xy[idx_temp][1]
                        if idx_temp == 0:

                            # if bbox_info_cp[idx_temp][0]['p1'][1] < cr_val['p1'][1]:
                            bbox_info_cp[idx][0] = {'p1': [int(cr_val['p1'][0] + idx_diff_cen_x),
                                                           int(cr_val['p1'][1] + idx_diff_cen_y)],
                                                    'p2': [int(cr_val['p2'][0] + idx_diff_cen_x),
                                                           int(cr_val['p2'][1] + idx_diff_cen_y)],
                                                    'p3': [int(cr_val['p3'][0] + idx_diff_cen_x),
                                                           int(cr_val['p3'][1] + idx_diff_cen_y)],
                                                    'p4': [int(cr_val['p4'][0] + idx_diff_cen_x),
                                                           int(cr_val['p4'][1] + idx_diff_cen_y)]}

                        else:

                            if bbox_info_cp[idx_temp][1]['p1'][1] < cr_val['p1'][1]:
                                 bbox_info_cp[idx][0] = {'p1': [int(cr_val['p1'][0] + diff_x),
                                                                int(cr_val['p1'][1] + diff_y)],
                                                         'p2': [int(cr_val['p2'][0] + diff_x),
                                                                int(cr_val['p2'][1] + diff_y)],
                                                         'p3': [int(cr_val['p3'][0] + diff_x),
                                                                int(cr_val['p3'][1] + diff_y)],
                                                         'p4': [int(cr_val['p4'][0] + diff_x),
                                                                int(cr_val['p4'][1] + diff_y)]}

            trans = []
            for ix,(cr_val,cr_key,keyword) in enumerate(bbox_info_cp):
                trans.append([cr_val['p1'][0] * self.resize_sh[1], cr_val['p1'][1] * self.resize_sh[0],
                              cr_val['p2'][0] * self.resize_sh[1], cr_val['p2'][1] * self.resize_sh[0],
                              cr_val['p3'][0] * self.resize_sh[1], cr_val['p3'][1] * self.resize_sh[0],
                              cr_val['p4'][0] * self.resize_sh[1], cr_val['p4'][1] * self.resize_sh[0]])

            return trans

        else:
            trans = []
            for info in self.bbox_info:
                # val, key = [], []
                # if self.endpoint == 'classify':
                #     val, key = info[0], info[1]

                # elif self.endpoint == 'document_form':
                val, key = info['sr_value'], info['sr_keyword']

                if val['p1'] == [0, 0] and val['p3'] == [0, 0]:

                    trans.append(
                        [(key['p1'][0] * self.resize_sh[1]) + test_kbox_list[0], (key['p1'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (key['p2'][0] * self.resize_sh[1]) + test_kbox_list[0], (key['p2'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (key['p3'][0] * self.resize_sh[1]) + test_kbox_list[0], (key['p3'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (key['p4'][0] * self.resize_sh[1]) + test_kbox_list[0], (key['p4'][1] * self.resize_sh[0]) + test_kbox_list[1]])

                else:
                    trans.append(
                        [(val['p1'][0] * self.resize_sh[1]) + test_kbox_list[0], (val['p1'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (val['p2'][0] * self.resize_sh[1]) + test_kbox_list[0], (val['p2'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (val['p3'][0] * self.resize_sh[1]) + test_kbox_list[0], (val['p3'][1] * self.resize_sh[0]) + test_kbox_list[1],
                         (val['p4'][0] * self.resize_sh[1]) + test_kbox_list[0], (val['p4'][1] * self.resize_sh[0]) + test_kbox_list[1]])

            return trans

    def make_insert_info(self,result_df,num_doc,test_img_cp):
        insert_info = []

        for idx, (area, data) in enumerate(zip(self.bbox_info, result_df)):
            k_bbox, v_bbox = area['sr_keyword'], area['sr_value']
            test_img_cp = drawimg_with_resize(self.resize_sh[1], self.resize_sh[0], k_bbox, v_bbox, test_img_cp)
            # if idx == 0:
            #     row = (num_doc, idx, str(json.dumps(area['sr_value'])), str(json.dumps(area['sr_value'])), area['keyword'], data[1], str(json.dumps(data[2])))
            #     insert_info.append(row)
            # else:
            row = (num_doc, idx, str(json.dumps(area['sr_keyword'])), str(json.dumps(area['sr_value'])), area['keyword'], data[1], str(json.dumps(data[2])))
            insert_info.append(row)

        return insert_info,test_img_cp

def responseCode(resultCode=9999,resultData='새로고침해줘',resultMsg=''):

    resultCode = int(resultCode)

    if resultCode == 200:
        resultData = resultData
    else:
        resultData = {'flag':resultData}

    result = {
              'resultCode':resultCode,
              'resultData': resultData,
              'resultMsg':str(resultMsg)
    }
    return Response(data=result)

def drawimg_with_resize(resz_w,resz_h,k_bbox,v_bbox,img):
    try:
        pts1 = [k_bbox['p1'][0] * resz_w, k_bbox['p1'][1] * resz_h, k_bbox['p2'][0] * resz_w, k_bbox['p2'][1] * resz_h,
               k_bbox['p3'][0] * resz_w, k_bbox['p3'][1] * resz_h, k_bbox['p4'][0] * resz_w, k_bbox['p4'][1] * resz_h]
        pts1 = np.array(pts1).reshape(-1, 2).astype(np.int32)
        img = cv2.polylines(img, [pts1], True, (0, 0, 255), 2)

        pts2 = [v_bbox['p1'][0] * resz_w, v_bbox['p1'][1] * resz_h, v_bbox['p2'][0] * resz_w, v_bbox['p2'][1] * resz_h,
               v_bbox['p3'][0] * resz_w, v_bbox['p3'][1] * resz_h, v_bbox['p4'][0] * resz_w, v_bbox['p4'][1] * resz_h]
        pts2 = np.array(pts2).reshape(-1, 2).astype(np.int32)
        img = cv2.polylines(img, [pts2], True, (0, 0, 255), 2)
    except:
        pass

    return img

def doc_clf(check_doc,doc_list):
    num_cls, doc_name = int(0), '미분류'  # num_doc of template image

    if len(check_doc) > 0:
        for doc in doc_list:

            if char_error_cnt2(doc[1], check_doc, return_type='per') >= 0.9:
                doc_name,num_cls = doc[1],int(doc[0])

    return num_cls,doc_name

def convert_to_text(text):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text).replace(" ", "")
    text = ''.join(text.split())
    return text

def same_label_merge(same_label,same_df):

    if same_label.shape[0] > 0:
        for idx, row in enumerate(same_label.iterrows()):
            min_idx = same_label[same_label['y1'] == min(same_label['y1'])]

            min_y4 = min_idx[min_idx['x1'] == min(min_idx['x1'])]['y4'].item()
            same_line = same_label[same_label['y1'] < min_y4]
            same_line = same_line.sort_values(by=['x1'], ascending=True)

            same_df = pd.concat([same_df, same_line])
            same_df = same_df.drop_duplicates(['text', 'x1', 'x4', 'y1', 'y4'])

            same_label = same_label.drop(same_label.index[0])

    return same_df

# def querytool():
#     try:
#         with connection.cursor() as con:
#             query = insert_tbl_document_form_query(table='base')
#             execute_values(con, query, insert_base, template=None, page_size=100)
#             num_doc = int(con.fetchone()[0])
#
#             insert_info, test_img_cp = api_model.make_insert_info(doc_area, result_df, num_doc, resz_w, resz_h,
#                                                                   test_img_cp)
#
#             query = insert_tbl_document_form_query(table='info')
#             execute_values(con, query, insert_info, template=None, page_size=100)
#     except Exception as e:
#         print('Error : ', e)
#         return responseCode(resultCode=500, resultData=e, resultMsg='DB Error')
#     finally:
#         if con:
#             con.close()
#     return

def delete_table_image(query1,value1,query2,value2,save_name,draw_file_name,error):
    try:
        with connection.cursor() as con:
            query1 = query1
            con.execute(query1, {'num_doc': value1})
            query2 = query2
            con.execute(query2, {'num_doc': value2})
    except Exception as e:
        print('Error : ', e)
        return responseCode(resultCode=500, resultData=str(e), resultMsg='DB Error')
    finally:
        if connection:
            connection.close()

    if os.path.isfile(settings.MEDIA_PATH + save_name):
        os.remove(settings.MEDIA_PATH + save_name)

    if os.path.isfile(settings.MEDIA_PATH + draw_file_name):
        os.remove(settings.MEDIA_PATH + draw_file_name)

    return responseCode(resultCode=500,resultData=error,resultMsg='분석실패')
