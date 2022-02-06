# -*- coding: utf-8 -*-

import sys
import os
import time
import requests
import argparse
import csv
import re
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from pytesseract import pytesseract
import cv2
from skimage import io
import numpy as np
from craft import craft_utils
from craft import imgproc
import craft.file_utils
import json
import zipfile
from craft.craft import CRAFT
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
from ocr.librarys import *
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.conf import settings
from django.views.static import serve
from django.db import connection

from craft import imgproc, text_detection
from librarys import *

@api_view(['POST'])
def cls(request):
    version,run_type= 'v3','demo'
    data = request.data.copy()
    doc_path = data['doc_path']
    userid = data['userid']

    sql = """SELECT nm_doc FROM tbl_document_form_base where userid=%(userid)s;"""
    cursor = connection.cursor()
    cursor.execute(sql, {'userid': userid})

    rows = cursor.fetchall()
    doc_list = []

    for row in rows:
        doc_list.append(row[0])

    print(doc_list)
    t = time.time()
    img_path = os.path.join(settings.MEDIA_PATH, doc_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    is_ok, ocr_data = post_request(img_path, end_point='ocr', add_data={"bboxes": 'true', "version": version, "run_type": run_type})
    # data = json.loads(ocr_data)
    result_list = {'bboxes':[],'ocr':[]}
    for i in ocr_data['data']:
        pts = i['vertices']
        result_list['bboxes'].append([pts['x1'],pts['y1'],pts['x2'],pts['y2'],pts['x3'],pts['y3'],pts['x4'],pts['y4']])
        result_list['ocr'].append(i['ocr'])
    # print(result_list['bboxes'])
    result_list, def_df = merge_predict2(result_list['bboxes'], result_list['ocr'],img)
    # print(result_list['ocr'].values)
    results = [result_list, def_df, img]

    check_docu = ''.join([q[-3] for q in def_df.values.tolist()]).replace(" ", "")
    check_docu = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', check_docu).replace(" ", "")
    # print(check_docu)
    # print(doc_list)
    docu_name = None
    for docu in doc_list:

        if char_error_cnt2(docu, check_docu, return_type='per') >= 0.8:
            docu_name = docu

            t_docu = time.time() - t
            print(f'time: {t_docu}')
            print(f'이 문서는 {docu} 입니다' + '\n')
            # end_point = None
            # myfile = {
            #     'name': docu_name
            # }
            # requests.request("POST", verify=False, url=settings.UI_URL + end_point + '/',files=myfile)

    query = """SELECT num_doc FROM tbl_document_form_base where nm_doc=%(nm_doc)s;"""
    cursor = connection.cursor()
    cursor.execute(query, {'nm_doc': docu_name})
    num_doc = cursor.fetchone()[0]
    print(num_doc)
    query = """SELECT doc_shape FROM tbl_document_form_base where nm_doc=%(nm_doc)s;"""
    cursor = connection.cursor()
    cursor.execute(query, {'nm_doc': docu_name})
    doc_sh = cursor.fetchone()[0]
    # print(doc_sh)
    doc_sh = list(map(int,re.sub('[{}]', '', doc_sh).split(',')))
    # print(doc_sh)
    bbox_list = {'key_box':[], 'val_box':[],'key_label':[]}
    query = """SELECT * FROM tbl_document_form_info where num_doc=%(num_doc)s;"""
    cursor = connection.cursor()
    cursor.execute(query, {'num_doc': num_doc})
    rows = cursor.fetchall()
    for row in rows:

        bbox_list['key_box'].append(eval(row[2]))
        bbox_list['val_box'].append(eval(row[3]))
        bbox_list['key_label'].append(row[-1])
    # print(bbox_list)
    t = time.time()

    arr_ocr = []
    for idx1, (i, text) in enumerate(zip(bbox_list["key_box"], bbox_list['key_label'])):
        if idx1 != 0:
            text_split = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text).replace(" ", "")
            for idx2, k in enumerate(results[0].values.tolist()):

                text2 = ''.join([q for q in k[-3]]).replace(" ", "")

                text2 = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text2).replace(" ", "")

                acc_cer2 = char_error_cnt(text_split, text2, return_type='per')

                if acc_cer2 > 0.7:
                    # print(text_split,text2)
                    test_kbox = k[0:8]
                    for ix, val in enumerate(bbox_list['key_label']):

                        val = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', val).replace(" ", "")

                        if text_split == val:

                            append_list = [[text, i, bbox_list['val_box'][ix]], [text2, test_kbox]]

                            if append_list not in arr_ocr:
                                arr_ocr.append(append_list)

    value_dic = {'label': [], 'detection': [], 'value_text': []}

    test_xy = results[1].iloc[:, 0:9]
    test_img = results[-1]

    # regist_doc = doc_list[1].split('=')[-1]
    regi_doc_path = os.path.join(settings.MEDIA_PATH, doc_path)
    regist_img = cv2.imread(regi_doc_path, cv2.IMREAD_COLOR)
    # print(regi_doc_path)
    point_rec_box_list = []
    # print(arr_ocr)
    for regist, test in arr_ocr:

        regist_h = (regist[1]['p4'][1] - regist[1]['p1'][1]) / doc_sh[0]

        point_rec_box = [
            [(test[1][0] / test_img.shape[1] + (regist[2]['p1'][0] - regist[1]['p1'][0]) / doc_sh[1]) *
             test_img.shape[1] - (regist_h / 2 * test_img.shape[0]), (
                         test[1][1] / test_img.shape[0] + (regist[2]['p1'][1] - regist[1]['p1'][1]) /
                         doc_sh[0] - regist_h / 4) * test_img.shape[0]],
            [(test[1][2] / test_img.shape[1] + (regist[2]['p2'][0] - regist[1]['p2'][0]) / doc_sh[1]) *
             test_img.shape[1] + (regist_h / 2 * test_img.shape[0]), (
                         test[1][3] / test_img.shape[0] + (regist[2]['p2'][1] - regist[1]['p2'][1]) /
                         doc_sh[0] - regist_h / 4) * test_img.shape[0]],
            [(test[1][4] / test_img.shape[1] + (regist[2]['p3'][0] - regist[1]['p3'][0]) / doc_sh[1]) *
             test_img.shape[1] + (regist_h / 2 * test_img.shape[0]), (
                         test[1][5] / test_img.shape[0] + (regist[2]['p3'][1] - regist[1]['p3'][1]) /
                         doc_sh[0] + regist_h / 4) * test_img.shape[0]],
            [(test[1][6] / test_img.shape[1] + (regist[2]['p4'][0] - regist[1]['p4'][0]) / doc_sh[1]) *
             test_img.shape[1] - (regist_h / 2 * test_img.shape[0]), (
                         test[1][7] / test_img.shape[0] + (regist[2]['p4'][1] - regist[1]['p4'][1]) /
                         doc_sh[0] + regist_h / 4) * test_img.shape[0]]
        ]

        point_rec_box_list.append(point_rec_box)

        box1 = test_xy[['x1', 'y1', 'x3', 'y3']]
        box2 = [point_rec_box[0][0], point_rec_box[0][1], point_rec_box[2][0], point_rec_box[2][1]]

        for ids, test_box in enumerate(box1.values):
            if IoU(test_box, box2) >= 0.9:
                xy = test_xy.values[ids].tolist()[0:8]
                if xy not in value_dic['detection']:
                    value_dic['label'].append(regist[0])
                    value_dic['detection'].append(xy)
                    value_dic['value_text'].append(test_xy.values[ids].tolist()[-1])

    test_img_cp = test_img.copy()
    result_df = pd.DataFrame(columns=['KEY', 'VALUE'])

    for klbl in bbox_list['key_label']:

        same_label = []

        for idx, lab in enumerate(value_dic['label']):
            if klbl == lab:

                pts = np.array(value_dic['detection'][idx]).reshape(-1, 2).astype(np.int32)
                test_img_cp = cv2.polylines(test_img_cp, [pts], True, (255, 0, 0), 2)

                if len(pts) > 0:
                    same_label.append(
                        [value_dic['value_text'][idx].split(' ')[0], pts[0][0], pts[3][0], pts[0][1],
                         pts[3][1]])

        same_label = pd.DataFrame(same_label, columns=['text', 'x1', 'x4', 'y1', 'y4'])
        same_df = pd.DataFrame(columns=['text', 'x1', 'x4', 'y1', 'y4'])
        if same_label.empty != True:
            for idx, row in enumerate(same_label.iterrows()):
                min_idx = same_label[same_label['y1'] == min(same_label['y1'])]

                min_y4 = min_idx[min_idx['x1'] == min(min_idx['x1'])]['y4'].item()
                same_line = same_label[same_label['y1'] < min_y4]
                same_line = same_line.sort_values(by=['x1'], ascending=True)

                same_df = pd.concat([same_df, same_line])
                same_df = same_df.drop_duplicates(['text', 'x1', 'x4', 'y1', 'y4'])

                same_label = same_label.drop(same_label.index[0])

        result_df = result_df.append({'KEY': klbl, 'VALUE': ' '.join([x[0] for x in same_df.values.tolist()])}, ignore_index=True)
    val_box = []
    for i in point_rec_box_list:

        pts2 = np.array(i).reshape(-1, 2).astype(np.int32)
        test_img_cp = cv2.polylines(test_img_cp, [pts2], True, (0, 255, 0), 2)

    t_3 = time.time() - t
    print(f'ocr2showtime: {t_3}')

    img_show(test_img_cp)
    print(result_df)

    return Response(data={'predict_result!':docu_name})
