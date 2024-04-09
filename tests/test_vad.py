import unittest
import numpy as np
from whisper_live.vad import VoiceActivityDetector


class TestVoiceActivityDetection(unittest.TestCase):
    def setUp(self):
        self.vad = VoiceActivityDetector()
        self.sample_rate = 16000

    def generate_silence(self, duration_seconds):
        return np.zeros(int(self.sample_rate * duration_seconds), dtype=np.float32)

    def test_vad_silence_detection(self):
        silence = self.generate_silence(3)
        is_speech_present = self.vad(silence.copy())
        self.assertFalse(
            is_speech_present, "VAD incorrectly identified silence as speech."
        )
