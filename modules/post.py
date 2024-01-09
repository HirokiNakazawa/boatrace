"""
LINE APIを使用して、予想結果を送信する
"""

import urllib.request
import json

url = "https://api.line.me/v2/bot/message/push"


def send_message(token: str = "", id: str = "", predict_list: list = []) -> None:
    """
    LINE APIを使用して予想結果をメッセージ送信する
    """
    header = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {token}"
    }

    message_list = []
    for predict_str in predict_list:
        message_list.append({
            "type": "text",
            "text": predict_str
        })

    req_data = json.dumps({
        "to": f"{id}",
        "messages": message_list
    })

    req = urllib.request.Request(
        url, data=req_data.encode(), method="POST", headers=header)

    try:
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
            headers = response.getheaders()
            status = response.getcode()

            # print(headers)
            # print(status)
            # print(body)
    except urllib.error.HTTPError as e:
        print(e.reason)
        # if e.code >= 400:
        #     error_body = json.loads(e.read())
        #     print(error_body)
