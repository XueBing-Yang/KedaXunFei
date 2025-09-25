import os
from turtle import mode
from dotenv import load_dotenv
import requests
import json
from openai import OpenAI
import json
from typing import List, Dict, Generator, Optional, Union
import websockets
import re
from loguru import logger


class ClModel():
    def __init__(self, temperature: float = 0.8, top_k: int = 10, max_tokens: int = 32784, url: str =  "http://10.43.107.29:9979/turing/v3/func/gpt", punish: int = 1):
        if url == None:
            self.url = os.getenv("CL_MODEL_URL")
        else:
            self.url = url
            
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.punish = punish
        
    def __call__(self, query: Optional[Union[str, List[Dict]]] = None):
        
        if isinstance(query, str):
            return self._chat_with_query(query)
        elif isinstance(query, List):
            return self._chat_with_messages(query)
    
    def _chat_with_query(self, query: str = None, stream: bool = False):
        request_data = {
            "header": {
                "traceId": "1"
            },
            "parameter": {
                "chat": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_k": self.top_k,
                    "stream": stream,
                    "punish": self.punish
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {
                            "content": query,
                            "role": "user"
                        }
                    ]
                }
            }
        }

        request_json = json.dumps(request_data)

        response = requests.post(self.url, data=request_json, headers={'Content-Type': 'application/json'})
        try:
            return json.loads(response.text)['payload']['choices']['text'][0]['content']
        except:
            logger.info(response)
    
    def _chat_with_messages(self, messages: List[Dict], stream: bool = False):
        request_data = {
            "header": {
                "traceId": "1"
            },
            "parameter": {
                "chat": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_k": self.top_k,
                    "stream": stream,
                    "punish": self.punish
                }
            },
            "payload": {
                "message": {
                    "text": messages
                }
            }
        }

        request_json = json.dumps(request_data)

        response = requests.post(self.url, data=request_json, headers={'Content-Type': 'application/json'})

        return json.loads(response.text)['payload']['choices']['text'][0]['content']

    async def stream_chat(self, messages: List[Dict] = None, stop: str = None, temperature: float = 0.9, max_tokens: int = 32784, top_k: int = 4):

        request_data = {
            "header": {
                "traceId": "SPARK_DEMO"
            },
            "parameter": {
                "chat": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_k": top_k
                }
            },
            "payload": {
                "message": {
                    "text": messages
                }
            }
        }

        async with websockets.connect(self.url) as websocket:
            await websocket.send(json.dumps(request_data))
            full_response = ""
            while True:
                response = await websocket.recv()
                response_data = json.loads(response)

                if response_data["header"]["code"] != 0:
                    logger.info(f"Error: {response_data['header']['message']}")
                    break

                for choice in response_data["payload"]["choices"]["text"]:
                    content = choice["content"]
                    full_response += content
                    if "<unused7>" in content:
                        contents = [content.split("<unused7>")[0], "<unused7>", content.split("<unused7>")[1]]
                        for content in contents:
                            if content == "": continue
                            yield content
                    else:
                        yield content

                if response_data["header"]["status"] == 2:
                    logger.info(f"response_data = {response_data}")
                    await websocket.close()
                    break
            
class VllmModel():
    def __init__(self, model_path: str = "gemini-2.5-pro", url: str = "https://www.chataiapi.com/v1"):
        if url == None:
            self.url = os.getenv("VLLM_BASE_URL")
            self.model_path = os.getenv("VLLM_MODEL_PATH")
        else:
            self.url = url
            self.model_path = model_path

        self.client = OpenAI(base_url=self.url, api_key="sk-nA6dH3q7Ky2ZTdNBAIKhTkUQkp1flOOWWo071bxPxS4AkAB1")

    def __call__(self, messages: Optional[Union[str, List[Dict]]] = None, only_content: bool = True, temperature: float = 0) -> str:

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        completion = self.client.chat.completions.create(
                                                model=self.model_path,
                                                messages = messages,
                                                max_tokens=16384,
                                                stream=False,
                                                temperature=temperature,
                                                )

        if only_content:
            return completion.choices[0].message.content
        return completion
    
    def stream_chat(self, messages: List[Dict] = None, only_content: bool = True) -> Generator[str, None, None]:
        completion = self.client.chat.completions.create(
                                        model=self.model_path,
                                        messages = messages,
                                        max_tokens=2048,
                                        stream=True,
                                        temperature=0.2,
                                        top_p=0.8
                                        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                if only_content:
                    yield chunk.choices[0].delta.content
                else:
                    yield chunk
        
if __name__ == "__main__":
    
    # import asyncio
    # model = ClModel(url='ws://10.43.107.29:9979/turing/v3/gpt')
    # messages = [{"role": "user", "content": "氮化镓的带隙宽度是多少？"}]
    # async def main():
    #     response = model.stream_chat(messages)
    #     async for res in response:
    #         print(res, end="", flush=True)

    # asyncio.run(main())
    
    
    query = "如何制备氮化镓"
    model = VllmModel()
    res = model(query)
    print(res)
    logger.info(res)


