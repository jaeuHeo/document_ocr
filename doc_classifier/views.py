# -*- coding: utf-8 -*-

import json
import random
import string

import django.conf
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from docx.shared import Pt, Mm
from urllib import parse
from PIL import Image
import os
import datetime
import time

from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.conf import settings
from django.views.static import serve
from django.db import connection

from .querys import *
from craft import imgproc
import librarys
from librarys import *
from ocr.librarys import *


@api_view(['GET'])
def get_alive_check(request):
    return Response(
        status=200,
        data = {
            "resultCode": 200,
            "resultMsg": "",
            "resultData": None
        }
    )

def get_paragraph(document, data):
    para = document.add_paragraph()
    paragraph_format = para.paragraph_format
    if 'top_margin' in data:
        paragraph_format.space_before = Mm(data['top_margin'])
    paragraph_format.space_after = 1

    paragraph_format.left_indent = Mm(data['left_indent'])
    return para

@api_view(['GET','PUT', "POST"])
def document_form(request):
    if request.method == 'PUT':
        return insert_document_form(request)

def insert_document_form(request):
    print('/doc_classifier/document_form')
    # print(request.data)
    data = request.data
    image, company_no = data.get('img_name', None), int(data.get('company_no', None))
    doc_info = json.loads(data['doc_info'])
    regist_sh = doc_info['sh']

    doc_area = json.loads(data['areas'])
    doc_area[0]['sr_keyword'] = doc_area[0]['sr_value']

    doc_name = doc_area[0]['keyword']

    api_model = librarys.api_model(endpoint='document_form', regist_sh=regist_sh, bbox_info=doc_area)

    save_name, local_name = api_model.get_img(image, company_no)
    save_file_name = 'DRAWIMG' + save_name
    img_path,test_img = api_model.load_img(save_name)
    print(img_path)

    test_img_cp = test_img.copy()
    test_sh = [test_img.shape[0], test_img.shape[1]]

    api_model.test_sh = test_sh
    api_model.resize_sh = [test_sh[0] / regist_sh[0], test_sh[1] / regist_sh[1]]

    version, run_type = 'v3', 'demo'
    try:
        is_ok, ocr_data = post_request(img_path, end_point='ocr', add_data={"bboxes": 'true', "version": version, "run_type": run_type})

    except Exception as e:
        print(f'GPU서버 에러, error = {e}')
        if os.path.isfile(settings.MEDIA_PATH + save_name):
            os.remove(settings.MEDIA_PATH + save_name)
        return responseCode(resultCode=500, resultData='GPU서버 에러', resultMsg=str(e))

    try:
        results = api_model.ocr_to_merge(ocr_data, test_img)

        result_df,test_img_cp = api_model.model(test_img_cp, results, num_test=0)
        result_df[0][0],result_df[0][1] = '문서 제목', result_df[0][0]

        info_list = [{'key':info[0],'value':info[1],'dbox':info[2]} for info in result_df]

    except Exception as e:
        print(f'분석실패, error: {e}')
        if os.path.isfile(settings.MEDIA_PATH + save_name):
            os.remove(settings.MEDIA_PATH + save_name)

        return responseCode(resultCode=500,resultData='분석실패',resultMsg='')

    conn = Connection()
    try:
        cur = conn.cursor()
        insert_base = [(doc_info['name'], local_name, str(list(regist_sh)), company_no)]
        query = insert_tbl_document_form_query(table='base')
        execute_values(cur, query, insert_base, template=None, page_size=100)
        num_doc = int(cur.fetchone()[0])

        insert_info, test_img_cp = api_model.make_insert_info(result_df, num_doc, test_img_cp)

        query = insert_tbl_document_form_query(table='info')
        execute_values(cur, query, insert_info, template=None, page_size=100)

        conn.commit()

    except Exception as e:
        print('postgresql database connection error!')
        print(e)
        if conn:
            conn.rollback()

        if os.path.isfile(settings.MEDIA_PATH + save_name):
            os.remove(settings.MEDIA_PATH + save_name)

        return responseCode(resultCode=500, resultData=e, resultMsg='db insert error')

    finally:
        if conn:
            conn.close()

    api_model.save_draw_image(test_img_cp, save_file_name)

    resultData = {'img_path': local_name, 'num_doc': num_doc, 'info_list': info_list}
    return responseCode(resultCode=200, resultData=resultData, resultMsg='')


@api_view(['POST'])
def show_info(request):
    data = request.data
    num_doc = data.get('num_doc', None)
    print(num_doc)

    conn = Connection()
    try:
        cur = conn.cursor()
        query1 = select_tbl_document_form_query(table='base')
        cur.execute(query1, {'num_doc': num_doc})
        rows_base = name_to_json(cur)
        doc_name = rows_base[0]['nm_doc']
        img_shape = json.loads(rows_base[0]['doc_shape'])
        img_path = rows_base[0]['doc_path']
        nm_img = img_path.split('=')[-1]

        query2 = select_tbl_document_form_query(table='info')
        cur.execute(query2, {'num_doc': num_doc})
        rows_info = name_to_json(cur)

        info_list,doc_json=[],[]
        for ids, row in enumerate(rows_info):
            info_dic,js_dic = {},{}
            info_dic['text_idx'] = ids+1
            info_dic['txt_key'] = row['txt_key']
            info_dic['valuetext'] = row['valuetext']
            if ids == 0:
                js_dic['titleText'] = row['txt_key']
            else:
                js_dic['keyText'] = row['txt_key']
                js_dic['valueText'] = row['valuetext']

            info_list.append(info_dic)
            doc_json.append(js_dic)
        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultMsg=e)

    finally:
        if conn:
            conn.close()

    save_name = settings.MEDIA_URL + '?img_name=' + 'DRAWIMG' + nm_img

    resultData ={'doc_list': info_list,'num_doc':num_doc, 'doc_name': doc_name, 'img_path': save_name, 'img_shape':img_shape, 'doc_json':doc_json}

    return responseCode(resultCode=200,resultData=resultData,resultMsg='')


@api_view(['POST'])
def recall(request):
    print('/doc_classifier/recall')

    data = request.data
    company_no = data['company_no']

    with Connection().cursor() as con:
        query = select_tbl_document_form_query_cno(table='base')
        con.execute(query, {'company_no': company_no})

        rows_base = name_to_json(con)

        info_list = []
        try:
            for row in rows_base:
                info_dic = {}
                info_dic['num_doc']=int(row['num_doc'])
                info_dic['doc_name']=row['nm_doc']
                split_strings = row['doc_path'].split('=')
                split_strings.insert(1, '=DRAWIMG')
                final_string = ''.join(split_strings)

                info_dic['doc_path']=final_string
                info_list.append(info_dic)

        except Exception as e:
            print(e)
            return responseCode(resultMsg=e)

    return responseCode(resultCode=200,resultData=info_list,resultMsg='')


@api_view(['POST'])
def correct(request):
    print('/doc_classifier/correct')
    data = request.data
    num_doc = data.get('num_doc', None)
    print(num_doc)

    conn = Connection()
    try:
        cur = conn.cursor()
        query1 = select_tbl_document_form_query(table='base')
        cur.execute(query1, {'num_doc': num_doc})
        rows_base = name_to_json(cur)
        doc_name,doc_path,regist_sh = rows_base[0]['nm_doc'],rows_base[0]['doc_path'].split('=')[-1],json.loads(rows_base[0]['doc_shape'])

        query2 = select_tbl_document_form_query(table='info')
        cur.execute(query2, {'num_doc': num_doc})
        rows_info = name_to_json(cur)

        info_list = []

        for ids, row in enumerate(rows_info):
            info_dic = {}
            if int(row['num_area']) == 0: info_dic['text_idx'] = '문서 제목'
            else: info_dic['text_idx'] = f'필드 {ids}'

            info_dic['txt_key'] = row['txt_key']
            info_dic['cr_key_area'] = json.loads(row['cr_key_area'])
            info_dic['cr_value_area'] = json.loads(row['cr_value_area'])

            info_list.append(info_dic)

        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultMsg=e)
    finally:
        if conn:
            conn.close()
    ### ?name=경로에다가 박스 안그린 이미지 띄울떄
    save_name = settings.MEDIA_URL + '?img_name=' + doc_path

    resultData={'doc_name':doc_name, 'doc_path':save_name, 'doc_shape':regist_sh, 'info_list':info_list}
    return responseCode(resultCode=200,resultData=resultData,resultMsg='')


@api_view(['PUT'])
def update_correct(request):
    print('/doc_classifier/update')
    data = request.data
    img_time = convert_to_text(''.join(str(datetime.datetime.now()).split(' ')))

    num_doc = data.get('num_doc', None)

    doc_info = json.loads(data['doc_info'])
    regist_sh = doc_info['sh']

    doc_area = json.loads(data['areas'])
    doc_area[0]['sr_keyword'] = doc_area[0]['sr_value']

    api_model = librarys.api_model(endpoint='document_form',regist_sh=regist_sh,bbox_info=doc_area)

    conn = Connection()
    try:
        cur = conn.cursor()
        query = select_tbl_document_form_query(table='base')
        cur.execute(query, {'num_doc': num_doc})

        rows_base = name_to_json(cur)

        image = rows_base[0]['doc_path'].split('=')[-1]

        company_no = rows_base[0]['company_no']

        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultCode=9999, resultData='다른 관리자가 템플릿을 삭제했습니다.', resultMsg=e)
    finally:
        conn.close()

    addtime_img = str(company_no) + '_' + img_time + '.' + image.split('.')[-1]
    addtime_path_doc = settings.MEDIA_URL + '?img_name=' + addtime_img
    file_oldname = os.path.join(settings.MEDIA_PATH, image)
    file_newname_newfile = os.path.join(settings.MEDIA_PATH, addtime_img)

    test_img = imgproc.loadImage(file_oldname)
    test_img_cp = test_img.copy()

    test_sh = [test_img.shape[0],test_img.shape[1]]
    api_model.test_sh = test_sh
    api_model.resize_sh = [test_sh[0]/regist_sh[0], test_sh[1]/regist_sh[1]]

    version, run_type = 'v3', 'demo'
    try:
        is_ok, ocr_data = post_request(file_oldname, end_point='ocr', add_data={"bboxes": 'true', "version": version, "run_type": run_type})
    except Exception as e:
        print(f'GPU서버 에러, error = {e}')
        return responseCode(resultCode=500, resultData='GPU서버 에러', resultMsg=e)

    try:
        results = api_model.ocr_to_merge(ocr_data,test_img)

        result_df, test_img_cp = api_model.model(test_img_cp, results, num_test=0)

        result_df[0][0], result_df[0][1] = '문서 제목', result_df[0][0]

        info_list = [{'key': info[0], 'value': info[1], 'dbox': info[2]} for info in result_df]

        insert_data, test_img_cp = api_model.make_insert_info(result_df,num_doc,test_img_cp)

    except Exception as e:
        print(e)

        return responseCode(resultCode=500,resultData='분석실패',resultMsg=e)

    conn = Connection()
    try:
        os.rename(file_oldname, file_newname_newfile)

        api_model.save_draw_image(test_img_cp, 'DRAWIMG' + addtime_img)

        if os.path.isfile(settings.MEDIA_PATH + 'DRAWIMG' + image):
            os.remove(settings.MEDIA_PATH + 'DRAWIMG' + image)

        cur = conn.cursor()
        query1 = delete_tbl_document_form_query(table='base')
        cur.execute(query1, {'num_doc': num_doc})

        info_data = [(num_doc, doc_info['name'], addtime_path_doc, str(list(regist_sh)), company_no)]
        query2 = insert_numdoc_nmdoc_path_shape_cno_tbl_document_form_base_query()
        psycopg2.extras.execute_values(cur, query2, info_data, template=None, page_size=100)

        query3 = delete_tbl_document_form_query(table='info')
        cur.execute(query3, {'num_doc': num_doc})

        query4 = insert_tbl_document_form_query(table='info')
        psycopg2.extras.execute_values(cur, query4, insert_data, template=None, page_size=100)

        # os.rename(file_oldname, file_newname_newfile)
        #
        # api_model.save_draw_image(test_img_cp, 'DRAWIMG' + addtime_img)
        #
        # if os.path.isfile(settings.MEDIA_PATH + 'DRAWIMG' + image):
        #     os.remove(settings.MEDIA_PATH + 'DRAWIMG' + image)

        conn.commit()
        resultData = {'info_list': info_list, 'doc_path': addtime_path_doc}
        return responseCode(resultCode=200, resultData=resultData, resultMsg='')

    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        # os.rename(file_newname_newfile, file_oldname)
        if os.path.isfile(settings.MEDIA_PATH + 'DRAWIMG' + addtime_img):
            os.remove(settings.MEDIA_PATH + 'DRAWIMG' + addtime_img)

        return responseCode(resultCode=9999, resultData='재시도', resultMsg=e)

    finally:
        if conn:
            conn.close()

    # try:
    #     os.rename(file_oldname, file_newname_newfile)
    #
    #     api_model.save_draw_image(test_img_cp, 'DRAWIMG' + addtime_img)

        # resultData = {'info_list': info_list, 'doc_path': addtime_path_doc}
        # return responseCode(resultCode=200, resultData=resultData, resultMsg='')

    # except Exception as e:
    #     print(e)
    #     if os.path.isfile(settings.MEDIA_PATH + 'DRAWIMG' + addtime_img):
    #         os.remove(settings.MEDIA_PATH + 'DRAWIMG' + addtime_img)

        # resultData = {'info_list': info_list, 'doc_path': file_newname_newfile}

        # return responseCode(resultCode=9999,resultData=resultData,resultMsg='')

    # if os.path.isfile(settings.MEDIA_PATH + 'DRAWIMG' + image):
    #     os.remove(settings.MEDIA_PATH + 'DRAWIMG' + image)
    # resultData = {'info_list': info_list, 'doc_path': addtime_path_doc}
    # resultData = {'info_list': info_list, 'doc_path': addtime_path_doc}
    # return responseCode(resultCode=200, resultData=resultData, resultMsg='')


@api_view(['PUT'])
def delete_image(request):
    print('doc_classifier/delete')
    data = request.data

    num_doc,type = data.get('num_doc', None),data.get('type', None)
    if num_doc is not None and type is not None:
        num_doc,type = int(num_doc),int(type)
    else:
        return responseCode(resultCode=500, resultData='num_doc,type 둘 중 하나가 없음',resultMsg='')

    conn = Connection()
    try:
        cur = conn.cursor()
        if type == 1:
            query1 = select_tbl_document_form_query(table='base')  # 템플릿이미지
            cur.execute(query1, {'num_doc': num_doc})
            rows_form_base = name_to_json(cur)
            path_doc_form = rows_form_base[0]['doc_path'].split('=')[-1]

            query2 = select_pathdoc_tbl_document_test_query()
            cur.execute(query2, {'num_cls': num_doc})
            rows_test_base = name_to_json(cur)

            path_doc_list = [row['path_doc'].split('=')[-1] for row in rows_test_base]

            path_doc_list.append(path_doc_form)

            ########transaction#########
            query3 = delete_tbl_document_form_query(table='info')

            cur.execute(query3, {'num_doc': num_doc})

            query4 = delete_tbl_document_form_query(table='base')
            cur.execute(query4, {'num_doc': num_doc})

            query5 = delete_tbl_document_test_num_cls_query(table='info')
            cur.execute(query5, {'num_cls': num_doc})

            query6 = delete_tbl_document_test_num_cls_query(table='base')
            cur.execute(query6, {'num_cls': num_doc})

            for path in path_doc_list:
                file_name = settings.MEDIA_PATH + path
                save_file_name = settings.MEDIA_PATH + 'DRAWIMG' + path
                if os.path.isfile(file_name):
                    os.remove(file_name)
                if os.path.isfile(save_file_name):
                    os.remove(save_file_name)

        elif type == 2:
            query = select_tbl_document_test_query(table='base')  # 테스트이미지

            cur.execute(query, {'num_doc': num_doc})
            rows_base = name_to_json(cur)
            path_doc = rows_base[0]['path_doc'].split('=')[-1]
            ########transaction#########
            query1 = delete_tbl_document_test_query(table='info')
            cur.execute(query1, {'num_doc': num_doc})

            query2 = delete_tbl_document_test_query(table='base')
            cur.execute(query2, {'num_doc': num_doc})

            file_name = settings.MEDIA_PATH + path_doc
            save_file_name = settings.MEDIA_PATH + 'DRAWIMG' + path_doc
            if os.path.isfile(file_name):
                os.remove(file_name)
            if os.path.isfile(save_file_name):
                os.remove(save_file_name)

        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultCode=9999,resultData='삭제 에러',resultMsg=e)
    finally:
        if conn:
            conn.close()

    return responseCode(resultCode=200, resultData={'num_doc':num_doc}, resultMsg='삭제완료')

@api_view(['POST'])
def clf(request):
    print(3213/0)
    # t = time.time()
    print('/doc_classifier/classify/')

    try:
        print('clf try')
        data = request.data
    except:
        print('clf except')
        data = request.data.copy()

    company_no = int(data['company_no'])
    is_web = data.get('is_web',False)
    doc_uuid = data.get('doc_uuid','')
    doc_path = data.get('img_name', None)

    api_model = librarys.api_model(endpoint='classify')
    try:
        save_name, local_name = api_model.get_img(doc_path, company_no)
    except Exception as e:
        print(e)
        return responseCode(resultCode=500, resultData='image save fault', resultMsg=str(e))

    print(save_name)
    save_file_name = 'DRAWIMG' + save_name
    doc_list = []

    with Connection().cursor() as con:
        query = select_tbl_document_form_query_cno(table='base')
        con.execute(query, {'company_no': company_no})
        rows = name_to_json(con)
        for row in rows:
            doc_list.append([row['num_doc'], row['nm_doc'], json.loads(row['doc_shape'])])

    try:
        img_path, test_img = api_model.load_img(save_name)
    except Exception as e:
        print(e)
        return responseCode(resultCode=500, resultData='File is not image', resultMsg=str(e))

    test_img_cp = test_img.copy()
    test_sh = [test_img.shape[0],test_img.shape[1]]
    api_model.test_sh = test_sh

    conn = Connection()
    try:
        cur = conn.cursor()
        insert_base = [(local_name, company_no, str(test_sh), doc_uuid, is_web)]

        query1 = insert_path_cno_sh_tbl_document_test_base_query()
        psycopg2.extras.execute_values(cur, query1, insert_base, template=None, page_size=100)
        num_test = cur.fetchone()[0]

        insert_info = [(int(num_test), int(0))]
        query2 = insert_num_idx_doc_tbl_document_test_info_query()
        psycopg2.extras.execute_values(cur, query2, insert_info, template=None, page_size=100)

        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultCode=500,resultData='db insert error',resultMsg=e)

    finally:
        if conn:
            conn.close()

    t = time.time()
    version, run_type = 'v3', 'demo'
    try:
        is_ok, ocr_data = post_request(img_path, end_point='ocr',add_data={"bboxes": 'true', "version": version, "run_type": run_type})
    except Exception as e:
        print(e)
        return responseCode(resultCode=500,resultData='gpu server error',resultMsg=e)

    print('gpu server time (ocr):', time.time()-t)

    results = api_model.ocr_to_merge(ocr_data, test_img)
    # results[1]['cen_y'] = results[1]['y1'] + (results[1]['y3'] - results[1]['y1']) / 2
    # results[1]['cen_x'] = results[1]['x1'] + (results[1]['x3'] - results[1]['x1']) / 2
    # results_sort = results[1].sort_values(by=["cen_y","cen_x"], ascending=[True,True])
    # min_cen_y = results[1][0]['cen_y']
    # for idx, row in results_sort.iterrows():
    #
    #     half_h = (row['y3']-row['y1'])/2
    #     step_df = results_sort.iloc[idx:,:]
    #
    #     same_line = step_df[(step_df['cen_y'] <= row['cen_y'] + half_h) & (step_df['cen_y'] >= row['cen_y'] - half_h)]
    #     print(same_line)
    #
    #     same_line = same_line.sort_values(by=['cen_x'], ascending=[True])
    #
    #     print('********')
    #     same_df = pd.concat([results_sort, same_line])
    #     same_df = same_df.drop_duplicates(['x1', 'y1', 'x3', 'y3'],keep='last')
    #     #
    #     # same_label = same_label.drop(same_label.index[0])
    #     print(same_df)
    #     print('next')
    check_doc = convert_to_text(''.join(results[1]['ocr']))
    print(check_doc)
    #######문서 분류#########
    t= time.time()

    num_cls,doc_name = doc_clf(check_doc,doc_list)

    print('classify time:', time.time()- t)
    ########################
    print(f'이 문서는 {doc_name} 입니다' + '\n')

    base_data = [(doc_name, int(num_cls), int(num_test))]
    conn = Connection()
    try:
        cur = conn.cursor()
        query = update_cls_num_tbl_document_test_base_query()
        cur.executemany(query, base_data)
        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultCode=500,resultData='db update error',resultMsg=e)
    finally:
        if conn:
            conn.close()

    time.sleep(3)

    if num_cls != int(0):
        for doc in doc_list:
            if doc[0] == int(num_cls):
                regist_sh = doc[2]
                api_model.regist_sh = regist_sh
                api_model.resize_sh = [test_sh[0]/regist_sh[0], test_sh[1]/regist_sh[1]]

    else:
        result_df = []

        if results[1].shape[0] > 0:
            for idx, q in enumerate(results[1].values.tolist()):
                pts = np.array(q[0:8]).reshape(-1, 2).astype(np.int32)
                test_img_cp = cv2.polylines(test_img_cp, [pts], True, (255, 0, 0), 2)
                result_df.append((num_test,idx,q[-3]))
        else:
            result_df = [(num_test,0,'')]

        api_model.save_draw_image(test_img_cp, save_file_name)

        conn = Connection()
        try:
            cur = conn.cursor()
            query1 = delete_tbl_document_test_query(table='info')
            cur.execute(query1, {'num_doc': int(num_test)})

            query2 = insert_num_doc_idx_txtval_tbl_document_test_info_query()
            psycopg2.extras.execute_values(cur, query2, result_df, template=None, page_size=100)

            conn.commit()
        except Exception as e:
            print(e)
            if conn:
                conn.rollback()
            query1 = delete_tbl_document_test_query(table='info')
            cur.execute(query1, {'num_doc': int(num_test)})
            query2 = delete_tbl_document_test_query(table='base')
            cur.execute(query2, {'num_doc': int(num_test)})
            conn.commit()
            return responseCode(resultCode=500,resultData='db delete and insert error',resultMsg=e)
        finally:
            if conn:
                conn.close()

        result_dic = {'class': doc_name, 'num_cls': num_cls, 'num_doc': int(num_test), 'path_doc': local_name,
                      'sh_doc': test_sh, 'txt_pairs': [{None: r[2]} for r in result_df]}

        return responseCode(resultCode=200, resultData=result_dic, resultMsg='complete')


    try:
        bbox_list = []
        with Connection().cursor() as con:
            query = select_tbl_document_form_query(table='info')
            con.execute(query, {'num_doc': int(num_cls)})
            rows_info = name_to_json(con)
            for idx, row in enumerate(rows_info):
                # if len(json.loads(row['cr_key_area'])) > 0: ######??#####
                if idx == 0:
                    bbox_list.append({'sr_value': json.loads(row['cr_value_area']), 'sr_keyword': json.loads(row['cr_value_area']), 'keyword': row['txt_key']})
                else:
                    bbox_list.append({'sr_value': json.loads(row['cr_value_area']), 'sr_keyword': json.loads(row['cr_key_area']), 'keyword': row['txt_key']})

        api_model.bbox_info = bbox_list

        result_df, test_img_cp = api_model.model(test_img_cp, results, num_test)
        # print(result_df)

        api_model.save_draw_image(test_img_cp, save_file_name)

    except Exception as e:
        print('e',e)
        conn = Connection()
        try:
            cur = conn.cursor()

            query1 = delete_tbl_document_test_query(table='base')
            cur.execute(query1, {'num_doc': num_test})
            query2 = delete_tbl_document_test_query(table='info')
            cur.execute(query2, {'num_doc': num_test})
            conn.commit()

        except Exception as e2:
            print('e2',e2)
            if conn:
                conn.rollback()
            return responseCode(resultData='테스트 이미지 삭제 실패', resultCode=500, resultMsg=e)
        finally:
            conn.close()

        if os.path.isfile(settings.MEDIA_PATH + save_name):
            os.remove(settings.MEDIA_PATH + save_name)
        if os.path.isfile(settings.MEDIA_PATH + save_file_name):
            os.remove(settings.MEDIA_PATH + save_file_name)

        return responseCode(resultData='분석실패',resultCode=500,resultMsg=e)

    conn = Connection()
    try:
        cur = conn.cursor()
        query1 = delete_tbl_document_test_query(table='info')
        cur.execute(query1, {'num_doc': int(num_test)})

        query2 = insert_numdoc_numidx_txtkeyval_tbl_document_test_info_query()
        psycopg2.extras.execute_values(cur, query2, result_df, template=None, page_size=100)

        conn.commit()
    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultCode=500,resultData='db delete and insert error', resultMsg=e)
    finally:
        if conn:
            conn.close()

    t_3 = time.time() - t
    print(f'after classify...and finish!: {t_3}')
    result_dic = {'class': doc_name, 'num_cls': int(num_cls), 'num_doc': int(num_test), 'path_doc': local_name,
                  'sh_doc': test_sh, 'txt_pairs': [{r[2]: r[3]} for r in result_df]}
    return responseCode(resultCode=200, resultData=result_dic, resultMsg='complete')


@api_view(['POST'])
def result(request):
    print('/doc_classifier/result/')
    data = request.data
    company_no = data['company_no']
    num_cls = data['num_cls']

    page = int(data['page']) + 10

    query = select_tbl_document_test_join_query()
    with connection.cursor() as con:
        con.execute(query, {'company_no': company_no})
        rows = con.fetchall()

    row_list = [[int(row[0]), row[2],row[3],json.loads(row[7]),row[5],row[6],json.dumps(row[8])] for row in rows]

    row_df = pd.DataFrame(row_list,columns=['num_doc', 'class', 'path_doc', 'shape_doc', 'txt_key', 'txt_val', 'num_cls'])
    row_df = row_df.groupby(['num_doc'],as_index=False).agg({'class':'first','path_doc':'first','shape_doc':'first','num_cls':'first','txt_key': lambda x:x.tolist(),'txt_val': lambda x:x.tolist()}).sort_index(ascending=False)

    res_list = []
    for idx,r in enumerate(row_df.values.tolist()):

        pair_dic = [{'txt_key': txt, 'txt_val': r[6][ids]} for ids,txt in enumerate(r[5])]

        if pair_dic[0]['txt_val'] is None:

            if r[4] != 'null': r[4] = int(r[4])
            else: r[4] = None

            res_list.append({'predicting':True,'num_doc': r[0],'num_cls': r[4], 'class': r[1], 'path_doc': r[2],'sh_doc': r[3], 'txt_pairs': pair_dic})

        elif pair_dic[0]['txt_val'] is not None and int(r[4]) == int(num_cls):
            nm_img = r[2].split('=')[-1]
            res_list.append({'predicting': False, 'num_doc': r[0], 'num_cls': int(r[4]),'class': r[1], 'path_doc': settings.MEDIA_URL + '?img_name=' +'DRAWIMG'+nm_img,'sh_doc': r[3], 'txt_pairs': pair_dic})

    return Response(data={'resultCode':200,'resultData': res_list,'resultMsg':''})

@api_view(['POST'])
def alldata(request):
    print('/doc_classifier/alldata/')
    data = request.data
    company_no = int(data['company_no'])

    conn = Connection()
    try:
        cur = conn.cursor()
        query = select_tbl_document_test_join_query()
        cur.execute(query, {'company_no': company_no})
        rows_join = name_to_json(cur)

        row_list = [[int(row['num_doc']), row['cls_doc'],row['path_doc'],json.loads(row['shape_doc']),row['txt_key'],row['txt_val'],json.dumps(row['num_cls']),row['doc_uuid'],row['is_web']] for row in rows_join]
        conn.commit()

    except Exception as e:
        print(e)
        if conn:
            conn.rollback()
        return responseCode(resultMsg=e)
    finally:
        if conn:
            conn.close()

    row_df = pd.DataFrame(row_list,columns=['num_doc', 'class', 'path_doc', 'shape_doc', 'txt_key', 'txt_val', 'num_cls','doc_uuid','is_web'])

    row_df = row_df.groupby(['num_doc'],as_index=False).agg({'class':'first','path_doc':'first','shape_doc':'first','num_cls':'first','txt_key': lambda x:x.tolist(),'txt_val': lambda x:x.tolist(),'doc_uuid':'first','is_web':'first'}).sort_index(ascending=False)

    res_list = []
    for idx,r in enumerate(row_df.values.tolist()):
        pair_dic = [{'txt_key': txt, 'txt_val': r[6][ids]} for ids, txt in enumerate(r[5])]
        nm_img = r[2].split('=')[-1]

        if r[4] != 'null' and pair_dic[0]['txt_val'] is None:
            # res_list.append({'predicting':True, 'num_doc': r[0], 'num_cls': int(r[4]),'class': r[1], 'path_doc': settings.MEDIA_URL + '?img_name=' +'DRAWIMG'+nm_img,'sh_doc': r[3], 'txt_pairs': pair_dic})
            res_list.append({'predicting': True, 'num_doc': r[0], 'num_cls': int(r[4]), 'class': r[1],
                             'path_doc': settings.MEDIA_URL + '?img_name=' + 'DRAWIMG' + nm_img, 'sh_doc': r[3],'txt_pairs': pair_dic,
                             'doc_uuid':r[7],'is_web': r[8]})


        elif r[4] != 'null' and pair_dic[0]['txt_val'] is not None:
            # res_list.append({'predicting':False, 'num_doc': r[0], 'num_cls': int(r[4]), 'class': r[1], 'path_doc': settings.MEDIA_URL + '?img_name=' + 'DRAWIMG' + nm_img, 'sh_doc': r[3],'txt_pairs': pair_dic})
            res_list.append({'predicting':False, 'num_doc': r[0], 'num_cls': int(r[4]), 'class': r[1], 'path_doc': settings.MEDIA_URL + '?img_name=' + 'DRAWIMG' + nm_img, 'sh_doc': r[3],'txt_pairs': pair_dic,
                             'doc_uuid':r[7],'is_web': r[8]})

    with Connection().cursor() as con:
        query = select_tbl_document_form_query_cno(table='base')
        con.execute(query, {'company_no': company_no})
        rows = name_to_json(con)

    class_list = [{'doc_name': row['nm_doc'], 'num_doc': int(row['num_doc'])} for row in rows]
    class_list.append({'doc_name':'미분류','num_doc':int(0)})

    return responseCode(resultCode=200,resultData={'class_list':class_list, 'testimg_list': res_list},resultMsg='')


@api_view(['POST','GET'])
def filter(request):
    print('/doc_classifier/filter/')

    if request.method == 'GET':
        data = request.query_params
    else:
        data = request.data

    type = int(data['type'])
    company_no = int(data['company_no'])
    search = parse.unquote(data['search'])
    search = f'%{search}%'
    page = int(data['page']) + 10

    res_list = []
    with Connection().cursor() as con:

        if type == 1:
            query = select_tbl_document_base_like_query(table='form')
            con.execute(query, {'search': search})
            rows_base = name_to_json(con)
            for row in rows_base:
                res_dic = {}

                if company_no == int(row['company_no']):
                    nm_img = row['doc_path'].split('=')
                    res_dic['num_doc'], res_dic['doc_name'] = int(row['num_doc']), row['nm_doc']

                    if res_dic['doc_name'] is not None:
                        res_dic['doc_path'] = nm_img[0] + '=DRAWIMG' + nm_img[-1]
                    else:
                        res_dic['doc_path'] = row['doc_path']
                    res_list.append(res_dic)

        elif type == 2:
            query = select_tbl_document_base_like_query(table='test')
            con.execute(query, {'search': search})
            rows_base = name_to_json(con)
            for row in rows_base:
                res_dic = {}

                if company_no == int(row['company_no']):
                    nm_img = row['path_doc'].split('=')
                    res_dic['num_doc'], res_dic['doc_name'] = int(row['num_doc']), row['cls_doc']

                    if res_dic['doc_name'] is not None:
                        res_dic['doc_path'] = nm_img[0] + '=DRAWIMG' + nm_img[-1]
                    else:
                        res_dic['doc_path'] = row['path_doc']
                    res_list.append(res_dic)

            # img_path = os.path.join(settings.SAVE_IMG_PATH, nm_img[-1])
            #
            # img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # if img is None:
            #     return Response(data={'resultCode':500,'resultData': '','resultMsg':f'저장된 {int(row[0])}번째 이미지 없음'})
            # res_dic['sh_doc'] = [img.shape[0],img.shape[1]]
            # res_dic['sh_doc'] = [1200, 800]


    # if type == 1:
    #     query = """SELECT * FROM tbl_document_form_info where num_doc = %(num_doc)s;"""
    # elif type == 2:
    #     query = """SELECT * FROM tbl_document_test_info where num_doc = %(num_doc)s;"""
    #
    # for idx,i in enumerate(res_list):
    #     pair_dic = []
    #
    #     cursor = connection.cursor()
    #     cursor.execute(query, {'num_doc': i['num_doc']})
    #     rows = cursor.fetchall()
    #     if type == 1:
    #         for row in rows:
    #             res_dic = {}
    #             if i['num_doc'] == int(row[0]):
    #                 res_dic['txt_key'] = row[4]
    #                 res_dic['txt_val'] = row[5]
    #                 pair_dic.append(res_dic)
    #     elif type == 2:
    #         for row in rows:
    #             res_dic = {}
    #             if i['num_doc'] == int(row[0]):
    #                 res_dic['txt_key']=row[2]
    #                 res_dic['txt_val']=row[3]
    #                 pair_dic.append(res_dic)
    #
    #     res_list[idx]['txt_pairs'] = pair_dic
    return responseCode(resultCode=200, resultData=res_list, resultMsg='')

@api_view(['POST'])
def exceldata(request):
    print('doc_classifier/exceldata')
    if request.method == 'GET':
        data = request.query_params
    else:
        data = request.data

    num_cls = data.get('num_cls', None)
    company_no = data['company_no']
    if num_cls is not None:
        num_cls = int(num_cls)
    else:
        return responseCode(resultCode=500,resultData=num_cls,resultMsg='num_cls가 없습니다.')

    with Connection().cursor() as con:
        query = select_tbl_document_form_query_cno(table='base')
        con.execute(query, {'company_no': company_no})
        rows_base = name_to_json(con)
        numdoc_list = [int(row['num_doc']) for row in rows_base]

    if num_cls not in numdoc_list:

        return responseCode()

    with Connection().cursor() as con:
        query = select_tbl_document_test_info_query()
        con.execute(query, {'num_cls': num_cls, 'company_no': company_no})
        rows_info = name_to_json(con)
        keyval_df = pd.DataFrame([[row['num_doc'],row['txt_key'],row['txt_val']] for row in rows_info],columns=['num_doc','key','val']).groupby(['num_doc']).agg({'key':lambda x:x.tolist(), 'val':lambda x:x.tolist()})

    keyval_list = [{k: v for k,v in zip(key,val)} for key,val in keyval_df.values]
    exl_df = pd.DataFrame(keyval_list).fillna('')
    exl_df.insert(0,'번호',exl_df.index + 1)
    header = exl_df.columns
    exl_df = exl_df.to_dict('records')

    return responseCode(resultCode=200, resultData={'data': exl_df, 'header':header}, resultMsg='')