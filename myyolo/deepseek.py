#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
DeepSeek语音助手 V2.0
基于官方文档重写，解决KWS问题
"""

import os
import threading
import base64
import hashlib
import hmac
import json
import time
import ssl
import configparser
import queue
import wave
from urllib.parse import urlencode
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time

import pyaudio
import websocket
from openai import OpenAI


# ==========================================================
# =================== 配置管理 =============================
# ==========================================================

def load_config():
    """加载配置文件"""
    config_file = 'config.ini'
    if not os.path.exists(config_file):
        print(f"错误：找不到配置文件 '{config_file}'")
        return None

    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    try:
        cfg = {
            'xf_appid': config.get('Xunfei', 'AppID'),
            'xf_api_key': config.get('Xunfei', 'APIKey'),
            'xf_api_secret': config.get('Xunfei', 'APISecret'),
            'dashscope_api_key': config.get('DashScope', 'APIKey'),
            'tts_vcn': config.get('Audio', 'TTS_VCN', fallback='aisjinger')
        }
        return cfg
    except (KeyError, configparser.NoSectionError) as e:
        print(f"错误: 配置文件 '{config_file}' 中缺少项: {e}")
        return None


# ==========================================================
# =================== 讯飞API基类 =========================
# ==========================================================

class XunfeiBase:
    """讯飞API基类"""

    def __init__(self, appid, api_key, api_secret):
        self.APPID = appid
        self.APIKey = api_key
        self.APISecret = api_secret

    def create_url(self, host, path):
        """生成WebSocket URL - 完全按照官方文档"""
        url = f'wss://{host}{path}'

        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": host
        }

        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        return url


# ==========================================================
# =================== 语音识别模块 =========================
# ==========================================================

class ASRModule(XunfeiBase):
    """讯飞语音识别模块（流式）"""

    def __init__(self, appid, api_key, api_secret):
        super().__init__(appid, api_key, api_secret)
        self.result_text = ""
        self.is_finished = threading.Event()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1600
        self.max_record_time = 10  # 最大录音时间（秒）
        self._temp_result = ""  # 临时存储每次分段结果

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            msg_json = json.loads(message)
            if msg_json.get('code') != 0:
                print(f"ASR Error: {msg_json.get('message')}")
                self.is_finished.set()
                return

            data = msg_json.get('data', {}).get('result', {})
            if data.get('ws'):
                result = ""
                for i in data['ws']:
                    for w in i.get('cw', []):
                        result += w.get('w', '')
                # 每次都累加到self.result_text
                self.result_text += result
                print(f"\rASR: {self.result_text}", end='', flush=True)

                if msg_json['data']['status'] == 2:
                    print()  # 换行
                    self.is_finished.set()
        except Exception as e:
            print(f"ASR on_message exception: {e}")

    def on_error(self, ws, error):
        """处理WebSocket错误"""
        print(f"ASR Error: {error}")
        self.is_finished.set()

    def on_close(self, ws, close_status_code, close_msg):
        """处理WebSocket关闭"""
        self.is_finished.set()

    def on_open(self, ws):
        """处理WebSocket连接建立"""

        def run(*args):
            # 发送第一帧，包含业务参数
            ws.send(json.dumps({
                "common": {"app_id": self.APPID},
                "business": {
                    "domain": "iat",
                    "language": "zh_cn",
                    "accent": "mandarin",
                    "vinfo": 1,
                    "vad_eos": 5000  # 5秒静音检测
                },
                "data": {
                    "status": 0,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw"
                }
            }))

            # 开始录音
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            print(">>> 正在录音，请说话...")
            start_time = time.time()

            try:
                while not self.is_finished.is_set():
                    # 检查是否超过最大录音时间
                    if time.time() - start_time >= self.max_record_time:
                        print(f"\n>>> 录音时间达到{self.max_record_time}秒，自动停止")
                        break

                    # 读取音频数据
                    audio_data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')

                    # 发送音频数据
                    ws.send(json.dumps({
                        "data": {
                            "status": 1,
                            "format": "audio/L16;rate=16000",
                            "audio": audio_base64,
                            "encoding": "raw"
                        }
                    }))

                    time.sleep(0.04)  # 40ms间隔

            finally:
                # 发送最后一帧
                ws.send(json.dumps({
                    "data": {
                        "status": 2,
                        "format": "audio/L16;rate=16000",
                        "audio": "",
                        "encoding": "raw"
                    }
                }))

                stream.stop_stream()
                stream.close()
                p.terminate()
                print(">>> 录音结束...")

        threading.Thread(target=run, daemon=True).start()

    def run(self):
        """运行语音识别"""
        self.result_text = ""
        self.is_finished.clear()

        ws_url = self.create_url(host="ws-api.xfyun.cn", path="/v2/iat")
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws.on_open = self.on_open

        # 在新线程中运行WebSocket
        threading.Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}, daemon=True).start()

        # 等待识别完成
        self.is_finished.wait()
        return self.result_text.strip()


# ==========================================================
# =================== DeepSeek对话模块 =====================
# ==========================================================

class DeepSeekModule:
    """DeepSeek对话模块"""

    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def chat(self, prompt):
        """与DeepSeek对话"""
        print(">>> 正在思考...")
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {
                        'role': 'system',
                        'content': '你是一个专业的中文语音助手，用户说什么你就直接专业简洁地回答什么，回答的问题在50字以内，不要重复用户问题，也不要寒暄。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                max_tokens=1000  # 限制输出长度
            )

            result = response.choices[0].message.content
            # 修复None返回值问题
            if result is None:
                return "抱歉，我没有理解您的问题，请重新提问。"
            return result.strip()

        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return "抱歉，这个问题有点难，您可以先问简单一些的问题吗？"


# ==========================================================
# =================== 语音合成模块 =========================
# ==========================================================

class TTSModule(XunfeiBase):
    """讯飞语音合成模块（流式播放）"""

    def __init__(self, appid, api_key, api_secret, vcn="xiaoyan"):
        super().__init__(appid, api_key, api_secret)
        self.vcn = vcn
        self.audio_queue = queue.Queue()
        self.pyaudio_instance = pyaudio.PyAudio()
        self.is_playing = threading.Event()

    def on_message(self, ws, message):
        """处理WebSocket消息"""
        try:
            message = json.loads(message)
            if message["code"] != 0:
                self.audio_queue.put(None)
                print(f"TTS Error: {message.get('message')}")
                return

            if "data" in message and "audio" in message["data"]:
                audio_data = base64.b64decode(message["data"]["audio"])
                self.audio_queue.put(audio_data)

            if message["data"]["status"] == 2:
                self.audio_queue.put(None)

        except Exception as e:
            print(f"TTS on_message exception: {e}")
            self.audio_queue.put(None)

    def on_error(self, ws, error):
        """处理WebSocket错误"""
        print(f"TTS Error: {error}")
        self.audio_queue.put(None)

    def on_close(self, ws, close_status_code, close_msg):
        """处理WebSocket关闭"""
        self.is_playing.set()

    def on_open(self, ws):
        """处理WebSocket连接建立"""
        # 发送合成请求
        ws.send(json.dumps({
            "common": {"app_id": self.APPID},
            "business": {
                "aue": "raw",
                "sfl": 1,
                "auf": "audio/L16;rate=16000",
                "vcn": self.vcn,
                "tte": "utf8"
            },
            "data": {
                "status": 2,
                "text": str(base64.b64encode(self.text.encode('utf-8')), "UTF8")
            }
        }))

    def _play_audio_thread(self):
        """音频播放线程"""
        stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            output=True
        )

        self.is_playing.clear()

        try:
            while True:
                audio_chunk = self.audio_queue.get()
                if audio_chunk is None:
                    break
                stream.write(audio_chunk)
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            self.is_playing.set()

    def run(self, text):
        """运行语音合成"""
        print(">>> 正在合成并播放语音...")
        self.text = text

        # 启动播放线程
        threading.Thread(target=self._play_audio_thread, daemon=True).start()

        # 建立WebSocket连接
        ws_url = self.create_url(host="tts-api.xfyun.cn", path="/v2/tts")
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        ws.on_open = self.on_open

        # 在新线程中运行WebSocket
        threading.Thread(target=ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}}, daemon=True).start()

        # 等待播放完成
        self.is_playing.wait()
        print(">>> 播放结束。")


# ==========================================================
# =================== 简单关键词检测 =======================
# ==========================================================

class SimpleKeywordDetector:
    """简单的关键词检测器（临时替代KWS）"""

    def __init__(self, keyword="你好问问"):
        self.keyword = keyword
        self.detected_event = threading.Event()
        self.is_running = False

    def start_listening(self):
        """开始监听关键词"""
        self.is_running = True
        self.detected_event.clear()

        print(f">>> 开始监听关键词: '{self.keyword}'")
        print(">>> 请说出唤醒词来开始对话...")

        # 简单的文本输入方式（临时方案）
        while self.is_running:
            try:
                user_input = input(">>> 请输入唤醒词或按Ctrl+C退出: ").strip()
                if user_input == self.keyword:
                    print(">>> 检测到唤醒词！")
                    self.detected_event.set()
                    break
                elif user_input.lower() in ['quit', 'exit', '退出']:
                    self.is_running = False
                    break
            except KeyboardInterrupt:
                print("\n>>> 用户中断程序")
                self.is_running = False
                break

    def wait_for_keyword(self):
        """等待关键词检测"""
        return self.detected_event.wait(timeout=1)

    def stop(self):
        """停止监听"""
        self.is_running = False


# ==========================================================
# =================== 主程序 ===============================
# ==========================================================

def main():
    """主程序"""
    print("=" * 50)
    print("DeepSeek语音助手 V2.0")
    print("=" * 50)

    # 加载配置
    cfg = load_config()
    if not cfg:
        print("配置加载失败，程序退出")
        return

    # 初始化模块
    asr_module = ASRModule(cfg['xf_appid'], cfg['xf_api_key'], cfg['xf_api_secret'])
    tts_module = TTSModule(cfg['xf_appid'], cfg['xf_api_key'], cfg['xf_api_secret'], cfg['tts_vcn'])
    llm_module = DeepSeekModule(cfg['dashscope_api_key'])
    keyword_detector = SimpleKeywordDetector("你好问问")

    print(">>> 语音助手初始化完成")
    print(">>> 唤醒词: '你好问问'")
    print(">>> 录音时间限制: 10秒")
    print(">>> 按Ctrl+C退出程序")
    print("-" * 50)

    try:
        while True:
            # 等待唤醒词
            print("\n>>> 等待唤醒词...")
            keyword_detector.start_listening()

            if not keyword_detector.is_running:
                break

            print("\n--- 检测到唤醒词，开始对话 ---")

            # 语音识别
            print(">>> 开始语音识别（最多10秒）...")
            user_text = asr_module.run()

            # 检查识别结果
            if not user_text or user_text.strip() == "":
                print(">>> 未识别到有效内容，请重新尝试")
                continue

            print(f">>> 识别结果: '{user_text}'")

            # AI对话
            print(f">>> 发送给DeepSeek: '{user_text}'")
            llm_response = llm_module.chat(user_text)
            print(f">>> DeepSeek回答: '{llm_response}'")

            # 语音合成
            if llm_response and llm_response.strip():
                tts_module.run(llm_response)
            else:
                print(">>> 没有有效的回答内容")

            print("\n--- 对话结束，等待下一次唤醒 ---")

    except KeyboardInterrupt:
        print("\n>>> 程序被用户中断，正在退出...")
    except Exception as e:
        print(f">>> 程序发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(">>> 程序退出")


if __name__ == "__main__":
    main()