import ast
import json
import os
import time
import openai
import pandas as pd
import requests
from tqdm import tqdm
import google.generativeai as genai

openai_client = openai.OpenAI(api_key="your_api_key")

gemini_api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=gemini_api_key, transport="rest")
google_client = genai.GenerativeModel('gemini-pro')


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=xxx&client_secret=xxx"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def Baidu_api(question):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k-0205?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": f"{question}"
            }
        ],
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload).json()
    time.sleep(1)
    return response["result"]


def Gemini_api(question):
    try:
        response = google_client.generate_content(contents=f'{question}', safety_settings=[
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ])
        return response.text
    except ValueError:
        print('ValueError')
        return "Sorry, I can't answer your question"



def GPT3_api(question):
    response = openai_client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user",
             "content": f"{question}"}
        ],
        temperature=1,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def GPT4_api(question):
    response = openai_client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=[
            {"role": "user",
             "content": f"{question}"}
        ],
        temperature=1,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def get_answer(LLM_name, api_func, question_csv_root, progress_file="GPT4_progress.txt"):
    question = pd.read_csv(question_csv_root, index_col=0)
    try:
        answer = pd.read_csv(f"{LLM_name}_answer.csv", index_col=0)
    except (FileNotFoundError, ValueError):
        answer = pd.DataFrame(columns=question.columns).to_csv(f"{LLM_name}_answer.csv",
                                                               index_label=question.index.name)

    # 尝试从进度文件中读取上次的进度
    try:
        with open(progress_file, "r") as f:
            start_index = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        start_index = 0

    for i, (index, type_question_series) in enumerate(tqdm(question.iterrows(), total=len(question))):
        if i < start_index:
            continue  # 跳过已经处理过的行

        type_answer = []
        for type_question in type_question_series:
            single_answer = api_func(type_question)
            type_answer.append(single_answer)
        answer.loc[index] = type_answer
        answer.to_csv(f"{LLM_name}_answer.csv")
        # 更新进度文件
        with open(progress_file, "w") as f:
            f.write(str(i + 1))



