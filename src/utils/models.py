import requests
import time
import openai
import os
from concurrent.futures import ThreadPoolExecutor

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def call_gpt_server(convo, model, max_tokens=512, depth_limit=3):
    try:
        if model == "o4-mini":
            response = client.chat.completions.create(
                        model= model,
                        messages = convo,
                        max_completion_tokens = max_tokens)
        else:
            response = client.chat.completions.create(
                            model= model,
                            messages = convo,
                            max_tokens = max_tokens)

        completion = response.model_dump()["choices"][0]["message"]["content"]
        return completion

    except Exception as e:
        if "content" in str(e):
            print(e)
            return None
        if "retry" in str(e):
            if depth_limit <= 10:
                time.sleep(2)
                return call_gpt_server(convo, model, max_tokens, depth_limit+1)
            else:
                return None
            
        else:
            print(e)
            return call_gpt_server(convo, model, max_tokens, depth_limit+1)

def batch_call_gpt_server(
    convos,
    model="gpt-4o",
    max_tokens=512,
    max_workers=50,
    depth_limit=3
):
    results = []

    count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(call_gpt_server, convo, model, max_tokens,depth_limit)
            for convo in convos
        ]

        for future in futures:
            output = future.result()
            results.append(output)
            count += 1
            print(f"Processed {count}/{len(convos)} conversations")

    return results

def call_open_llm_api(convo, model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", max_tokens=2048, depth_limit=3, host="http://localhost:8001"):
    url = f"{host}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": convo,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        completion = result["choices"][0]["message"]["content"]
        return completion

    except Exception as e:
        print(f"Error: {e}")

        if depth_limit <= 10:
            time.sleep(2)
            return call_open_llm_api(convo, model, max_tokens, depth_limit+1, host)
        else:
            return None


def batch_call_open_llm_api(
    convos, 
    max_tokens=512, 
    host="http://localhost:8001", 
    max_workers=5,
    model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    depth_limit=3
):

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(call_open_llm_api, convo, model,  max_tokens, depth_limit, host)
            for convo in convos
        ]

        for future in futures:
            output = future.result()
            results.append(output)

    return results
