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