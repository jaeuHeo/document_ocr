# -*- coding: utf-8 -*-

# import json
#
# import random
# import string
import numpy as np
import pandas as pd
# import psycopg2
# from psycopg2.extras import execute_values
# from docx.shared import Pt, Mm
# from urllib import parse
# from PIL import Image
# import os
# import datetime
# import time
# from rest_framework.decorators import api_view
# from rest_framework import status
# from rest_framework.response import Response
# from django.conf import settings
# from django.views.static import serve
# from django.db import connection
import re
# # from .querys import *
# from craft import imgproc
# from librarys import *
# from ocr.librarys import *

# same_label = [['da',1,1,1],['aa',2,2,2],['ac',3,3,3]]
# same_df = pd.DataFrame(same_label, columns=['text','x1','y1','x2'])
# a = ' '.join([x[0] for x in same_df.values])
# print(a)
# text = 'wadaw d12..dwd4 호쟈우'
# text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
# text = ''.join(text.split())
# print(text)

# df = pd.DataFrame({
#     'name':
#     ['orange','banana','lemon','mango','apple'],
#     'price':
#     [2,3,7,21,11],
#     'stock':
#     ['Yes','No','Yes','No','Yes']
# })
# print((1+1+1+0.67+1+1+0.8+1+1+0.86+1+1+0.8+1+0.83)/17)
# for idx,i in enumerate(df):
#     print([df.iloc[idx,0:2].values])
# ['ㅅ', 'ㅏ', 'ㅇ', 'ㅓ', 'ㅂ', 'ㅈ', 'ㅏ', 'ㄷ', 'ㅡ', 'ㅇ', 'ㄹ', 'ㅗ', 'ㄱ', 'ㅈ', 'ㅡ', 'ㅇ'] ['ㅎ', 'ㅓ', 'ㅈ', 'ㅐ', 'ㅇ', 'ㅜ']
# sample_text = ['ㅅ', 'ㅏ', 'ㅇ', 'ㅓ', 'ㅂ', 'ㅈ', 'ㅏ', 'ㄷ', 'ㅡ', 'ㅇ', 'ㄹ', 'ㅗ', 'ㄱ', 'ㅈ', 'ㅡ', 'ㅇ']
# a = np.array(sample_text)
# s_decompose_text = ['ㅎ', 'ㅓ', 'ㅈ', 'ㅐ', 'ㅇ', 'ㅜ']
# b = np.array(s_decompose_text)
# print(np.array(sample_text),np.array(s_decompose_text))
# print((np.array(sample_text) == np.array(s_decompose_text)).sum())
# def tbl_document_query(method,col,form,base):
#     if method == 'select':
#         query = 'select ' + col + ' from' ' tbl_document_' + form +'_'+ base
#         return query
#
# print(tbl_document_query('select','(21,22)','form','base'))
aa=[1,2]
a,b = aa
print(a,b)