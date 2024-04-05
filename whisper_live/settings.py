import enum
from typing import List, Tuple, Union, Optional, Iterable
from faster_whisper.vad import VadOptions


# 다음을 수정하여 Whisper Transcriber의 설정을 변경할 수 있습니다.
whisper_model = "large-v3"
# whisper_model = "small"
whisper_parameters = {
    "language": "ko",
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "no_speech_threshold": 0.6,
    "without_timestamps": False,
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {
        "threshold": 0.6,  # threshold: 말하기 임계값입니다. Silero VAD는 각 오디오 청크에 대한 말하기 확률을 출력하며, 이 값이 임계값을 초과하면 SPEECH(음성)로 간주됩니다. 각 데이터셋에 대해 이 매개변수를 별도로 조정하는 것이 좋지만, 대부분의 데이터셋에 대해 "게으른" 0.5가 꽤 좋은 성능을 보입니다.
        "min_speech_duration_ms": 250,  # 최소 말하기 지속 시간(밀리초)입니다. min_speech_duration_ms보다 짧은 최종 음성 청크는 제거됩니다.
        "max_speech_duration_s": float(
            "inf"
        ),  # 말하기 청크의 최대 지속 시간(초)입니다. max_speech_duration_s보다 긴 청크는 100ms 이상 지속되는 마지막 침묵의 타임스탬프에서 분할되어 공격적인 절단을 방지합니다. 그렇지 않으면 max_speech_duration_s 직전에 공격적으로 분할됩니다.
        "min_silence_duration_ms": 2000,  # 각 말하기 청크의 끝에서 min_silence_duration_ms 동안 기다린 후 분리합니다.
        "window_size_samples": 1024,  # window_size_samples 크기의 오디오 청크가 silero VAD 모델에 공급됩니다. 경고! Silero VAD 모델은 16000 샘플 레이트에 대해 512, 1024, 1536 샘플을 사용하여 훈련되었습니다. 이 값들 외의 다른 값은 모델 성능에 영향을 줄 수 있습니다!!
        "speech_pad_ms": 400,  # 최종 말하기 청크는 각면에 speech_pad_ms만큼 패딩됩니다.
    },
    "chunk_length": None,
    "clip_timestamps": "0",
    # "hallucination_silence_threshold": 1.0,
}

if not whisper_parameters.get("vad_filter"):
    whisper_parameters["vad_parameters"] = None


class WhisperConfig(enum.Enum):
    task: str = ("transcribe",)
    beam_size: int = (
        1,
    )  ### 빔 크기가 클수록 탐색 공간이 넓어져 더 정확한 결과를 얻을 수 있지만, 계산 비용이 증가한다.
    best_of: int = (1,)  ### 빔 탐색에서 반환되는 최상의 후보 수를 결정한다.
    patience: float = (
        1,
    )  ### 결과를 기다리는 시간을 조정함. 더 높은 시간은 더 오래 기다림을 의미한다.
    length_penalty: float = (1,)  # 출력 길이에 대한 패널티
    repetition_penalty: float = (1,)  # 반복되는 단어나 구절에 대한 패널티
    no_repeat_ngram_size: int = (0,)  ### 반복을 피하기 위해 고려할 n-gram 크기
    temperature: Union[float, List[float], Tuple[float, ...]] = (
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )  ### 생성된 텍스트의 다양성을 제어한다. 0에 가까울수록 확신 있는 출력을 생성하고, 1에 가까울수록 더 다양한 출력을 생성함
    compression_ratio_threshold: Optional[float] = (
        2.4,
    )  # 압축 비율 임계값을 설정하여 너무 길거나 짧은 출력을 걸러낸다.
    log_prob_threshold: Optional[float] = (
        -1.0,
    )  # 로그 확률 임계값을 설정하여 너무 낮은 확률의 출력을 걸러낸다.
    no_speech_threshold: Optional[float] = (
        0.6,
    )  # 말이 없는 것으로 간주되는 임계값을 설정함
    condition_on_previous_text: bool = (True,)  # 이전 텍스트를 고려할지 여부를 결정함
    prompt_reset_on_temperature: float = (
        0.5,
    )  # 특정 온도에서 프롬푸트를 재설정할지 여부를 결정함
    initial_prompt: Optional[Union[str, Iterable[int]]] = (
        None,
    )  # 초기 프롬프투 설정을 위한 문자열 또는 토큰 ID
    prefix: Optional[str] = (None,)  # 모든 출력에 앞서 붙일 접두사 문자열
    suppress_blank: bool = (True,)  # 빈 토큰을 억제할지 여부를 결정함
    suppress_tokens: Optional[List[int]] = ([-1],)  # 제거할 토큰 ID 목록
    without_timestamps: bool = (False,)  # 출력에 타임스탬프를 포함할지 여부를 결정함
    max_initial_timestamp: float = (1.0,)  # 초기 타임스탬프의 최대 값
    word_timestamps: bool = (False,)  # 단어별 타임스탬프를 생성할지 여부를 결정함
    prepend_punctuations: str = ("\"'“¿([{-",)
    append_punctuations: str = ("\"'.。,，!！?？:：”)]}、",)
    vad_filter: bool = (False,)  # 음성 활동 탐지 필터를 사용할지 여부를 결정함
    vad_parameters: Optional[Union[dict, VadOptions]] = (None,)  # VAD 필터의 매개변수
    max_new_tokens: Optional[int] = (None,)  # 생성할 최대 토큰 수
    chunk_length: Optional[int] = (
        None,
    )  ### 처리할 오디오 청크의 길이. 길이가 길수록 더 많은 메모리가 필요하지만, 더 정확한 결과를 얻을 수 있다.
    clip_timestamps: Union[str, List[float]] = ("0",)  # 타임스탬프를 자를 위치
    hallucination_silence_threshold: Optional[float] = (
        None,
    )  # 환각적인 소리를 제거하기 위한 임계값
