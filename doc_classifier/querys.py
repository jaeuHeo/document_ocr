from django.db import connection

def select_tbl_document_form_query(table):
    if table == 'base':
        return """SELECT * FROM tbl_document_form_base where num_doc=%(num_doc)s order by num_doc ASC;"""
    elif table == 'info':
        return """SELECT * FROM tbl_document_form_info where num_doc=%(num_doc)s order by num_doc ASC, num_area ASC;"""

def select_tbl_document_form_query_cno(table):
    if table == 'base':
        return """SELECT * FROM tbl_document_form_base where company_no=%(company_no)s order by num_doc ASC;"""
    elif table == 'info':
        return """SELECT * FROM tbl_document_form_info where company_no=%(company_no)s order by num_doc ASC, num_area ASC;"""

def select_pathdoc_tbl_document_test_query():
    return """SELECT path_doc FROM tbl_document_test_base where num_cls=%(num_cls)s order by num_doc ASC;"""

def select_tbl_document_test_query(table):
    if table == 'base':
        return """SELECT * FROM tbl_document_test_base where num_doc=%(num_doc)s order by num_doc ASC;"""
    elif table == 'info':
        return """SELECT * FROM tbl_document_test_info where num_doc=%(num_doc)s order by num_doc ASC, num_area ASC;"""

def select_tbl_document_test_join_query():
    return """
        SELECT
            t1.num_doc,
            t2.idx_doc,
            t1.cls_doc,
            t1.path_doc,
            t1.company_no,
            t2.txt_key,
            t2.txt_val,
            t1.shape_doc,
            t1.num_cls,
            t1.doc_uuid,
            t1.is_web
        FROM
            (
            SELECT
                *
            FROM
                tbl_document_test_base
            WHERE
                company_no = %(company_no)s
            ORDER BY
                num_doc) AS t1
        INNER JOIN tbl_document_test_info AS t2 ON
            t1.num_doc = t2.num_doc
        ORDER BY
            t2.num_doc DESC,
            t2.idx_doc ASC;"""

def select_distinct_keytext_query():
    return """SELECT DISTINCT txt_key from tbl_document_test_info where num_doc in (select num_doc from tbl_document_test_base where (num_cls,company_no) = (%(num_cls)s,%(company_no)s));"""

def select_tbl_document_test_info_query():
    return """SELECT num_doc,txt_key,txt_val from tbl_document_test_info where num_doc in (select num_doc from tbl_document_test_base where (num_cls,company_no) = (%(num_cls)s,%(company_no)s)) ORDER BY num_doc ,idx_doc;"""

def select_tbl_document_base_like_query(table):
    if table == 'form':
        return """SELECT * FROM tbl_document_form_base WHERE nm_doc LIKE (%(search)s) ORDER BY num_doc ASC;"""
    elif table == 'test':
        return """SELECT * FROM tbl_document_test_base WHERE cls_doc LIKE (%(search)s) ORDER BY num_doc ASC;"""
def insert_imgfile_base_query():
    return """insert into tbl_document_form_base(file) values (%s);"""

def insert_tbl_document_form_query(table='base'):
    if table == 'base':
        return """insert into tbl_document_form_base(nm_doc,doc_path,doc_shape,company_no) values %s RETURNING num_doc;"""
    elif table == 'info':
        return """insert into tbl_document_form_info(num_doc, num_area, cr_key_area, cr_value_area, txt_key,valuetext,detectionbox) values %s;"""

def insert_tbl_document_form_info_query():
    return """insert into tbl_document_form_info(num_doc, num_area, cr_key_area, cr_value_area, txt_key,valuetext,detectionbox) values %s;"""

def insert_path_cno_sh_tbl_document_test_base_query():
    return """INSERT INTO tbl_document_test_base(path_doc,company_no,shape_doc,doc_uuid,is_web) values %s RETURNING num_doc;"""

def insert_num_idx_doc_tbl_document_test_info_query():
    return """INSERT INTO tbl_document_test_info(num_doc,idx_doc) values %s;"""

def insert_numdoc_nmdoc_path_shape_cno_tbl_document_form_base_query():
    return """insert into tbl_document_form_base(num_doc, nm_doc, doc_path, doc_shape, company_no) values %s;"""

def insert_numdoc_numidx_txtkeyval_tbl_document_test_info_query():
    return """INSERT INTO tbl_document_test_info(num_doc,idx_doc,txt_key,txt_val) values %s ;"""

def insert_num_doc_idx_txtval_tbl_document_test_info_query():
    return """INSERT INTO tbl_document_test_info(num_doc,idx_doc,txt_val) values %s ;"""

def update_cls_num_tbl_document_test_base_query():
    return """UPDATE tbl_document_test_base SET (cls_doc,num_cls) = (%s,%s) WHERE num_doc = %s;"""

def update_tbl_document_form_info_query():
    return """UPDATE tbl_document_form_info SET (num_area,cr_key_area,cr_value_area,txt_key) = (%s,%s,%s,%s) where num_doc=(%s);"""

def delete_tbl_document_test_num_cls_query(table):
    if table == 'base':
        return """delete from tbl_document_test_base where num_cls = %(num_cls)s;"""
    elif table =='info':
        return """delete from tbl_document_test_info where num_doc in (select num_doc from tbl_document_test_base where num_cls = %(num_cls)s);"""

def delete_tbl_document_test_query(table):
    query = ''

    if table == 'base':
        query = """DELETE FROM tbl_document_test_base WHERE num_doc = %(num_doc)s;"""
    elif table == 'info':
        query = """DELETE FROM tbl_document_test_info WHERE num_doc = %(num_doc)s;"""

    return query

def delete_tbl_document_form_query(table):
    query = ''

    if table == 'base':
        query = """DELETE FROM tbl_document_form_base WHERE num_doc = %(num_doc)s;"""
    elif table == 'info':
        query = """DELETE FROM tbl_document_form_info WHERE num_doc = %(num_doc)s;"""

    return query



def tbl_document_form_query(method, Command_col,constraint_col,constraint_val):
    rows = []
    if method == 'select':

        query = """SELECT %(Command_col)s FROM tbl_document_form_base where %(constraint_col)s=%(constraint_val)s;"""
        with connection.cursor() as con:
            con.execute(query, {'Command_col':Command_col,'constraint_col': constraint_col,'constraint_val':constraint_val})
            rows = con.fetchall()

    elif method == 'update':

        query = """UPDATE tbl_document_form_base SET (num_area,cr_key_area,cr_value_area,txt_key) = (%s,%s,%s,%s) where num_doc=(%s);"""
        with connection.cursor() as con:
            con.execute(query, {'Command_col': Command_col, 'constraint_col': constraint_col, 'constraint_val': constraint_val})
            rows = con.fetchmany()

    return rows