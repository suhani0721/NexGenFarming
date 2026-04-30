import os
import base64
import re
from gtts import gTTS
from dotenv import load_dotenv
from langsmith import traceable
from io import BytesIO

load_dotenv()


class TTSModel:
    def __init__(self):
        self.prompt = """
Speak in Hindi.
Use simple Indian rural Hindi.
Speak in a calm, friendly, and supportive tone.
Sound like an experienced female agricultural advisor guiding a farmer.
Maintain clear pronunciation and moderate speaking speed.
Pause slightly between steps for clarity.
Avoid sounding robotic or dramatic.
"""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"[#*`>-]", "", text)
        text = re.sub(r"\d+\.", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    @traceable(name="Chrome Hindi TTS Generation")
    def synthesize(self, text: str):
        cleaned_text = self.clean_text(text)
        full_text = self.prompt + "\n\n" + cleaned_text

        try:
            # ✅ Generate Hindi audio (Chrome-like voice)
            tts = gTTS(text=full_text, lang='hi', tld='co.in', slow=False)

            # Save to memory instead of file
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_bytes = audio_buffer.getvalue()

            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return audio_base64

        except Exception as e:
            print("TTS ERROR:", str(e))
            return None