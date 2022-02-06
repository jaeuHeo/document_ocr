import psycopg2
import json

from psycopg2.extras import execute_values



class Connection:
    def cursor(self):
        conn = psycopg2.connect(host='172.16.114.128', dbname='doc_classifier', user='postgres', password='postgres')
        cur = conn.cursor()
        return cur

def name_to_json(cursor):
    """
    cursor.fetchall() 함수로 받아온 쿼리 결과를 json 형식으로 만들어 반환해주는 함수입니다.
    :param cursor: SQL 연결 변수
    :return: JSON 쿼리 결과 LIST
    """
    row = [dict((cursor.description[i][0], value)
                for i, value in enumerate(row)) for row in cursor.fetchall()]
    return row


def json_result(status=None, data=None, message=None):
    """
    response format
    :param status: status code
    :param data: response data
    :param message: response message
    :return: JSON
    """
    result = {
        'code': status,
        'message': message,
        'data': data
    }

    return result

connection = Connection()
cursor = connection.cursor()

data = {
    "doc_info":{
        'name':'사업자등록증 양식'
    },
    'title':{
        'keyword':'사업자등록증',
        'sr':{"p1": [1, 2], "p2": [1, 2], "p3": [1, 2], "p4": [1, 2]}
    },
    'areas':[
        {
            'keyword':'사업자번호',
            'sr_keyword':{"p1": [1, 2], "p2": [1, 2], "p3": [1, 2], "p4": [1, 2]},
            'sr_value':{"p1": [1, 2], "p2": [1, 2], "p3": [1, 2], "p4": [1, 2]}
         },
        {
            'keyword':'업종',
            'sr_keyword':{"p1": [1, 2], "p2": [1, 2], "p3": [1, 2], "p4": [1, 2]},
            'sr_value':{"p1": [1, 2], "p2": [1, 2], "p3": [1, 2], "p4": [1, 2]}
         }
    ]

}

doc_info = data['doc_info']
### 문서 등록
sql = """insert into tbl_document_form_base(nm_doc) values (%(doc_name)s) RETURNING num_doc;"""

cursor.execute(sql, {'doc_name':doc_info['name']})
doc_num = cursor.fetchone()[0]

data['title']

title_info = data['title']
areas_info = data['areas']

insert_data = []
insert_data.append(tuple([doc_num, 0, json.dumps(title_info['sr']), None, title_info['keyword']]))
for idx, area in enumerate(areas_info):
    row = (doc_num, idx+1, json.dumps(area['sr_keyword']), json.dumps(area['sr_value']), area['keyword'])
    insert_data.append(row)

sql = """insert into tbl_document_form_info(num_doc, num_area, cr_key_area, cr_value_area, txt_key) values %s;"""

psycopg2.extras.execute_values (
    cursor, sql, insert_data, template=None, page_size=100
)
cursor.execute('commit;')

cursor.close()


