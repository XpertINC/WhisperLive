import enum
from typing import List, Tuple, Union, Optional, Iterable
from faster_whisper.vad import VadOptions


# 다음을 수정하여 Whisper Transcriber의 설정을 변경할 수 있습니다.
whisper_parameters = {
    "task": "transcribe",
    "beam_size": 1,
    "best_of": 1,
    "patience": 1,
    "no_speech_threshold": 0.6,
    "without_timestamps": False,
    "word_timestamps": False,
    "word_timestamps": False,
    "vad_filter": True,
    "vad_parameters": {"threshold": 0.5},
    "chunk_length": None,
    "clip_timestamps": "0",
    "hallucination_silence_threshold": None
}

if not whisper_parameters.get("val_filter"):
    whisper_parameters["vad_parameters"] = None
    

class WhisperConfig(enum.Enum):
    task: str = "transcribe",
    beam_size: int = 1, ### 빔 크기가 클수록 탐색 공간이 넓어져 더 정확한 결과를 얻을 수 있지만, 계산 비용이 증가한다.
    best_of: int = 1, ### 빔 탐색에서 반환되는 최상의 후보 수를 결정한다. 
    patience: float = 1, ### 결과를 기다리는 시간을 조정함. 더 높은 시간은 더 오래 기다림을 의미한다.
    length_penalty: float = 1, # 출력 길이에 대한 패널티
    repetition_penalty: float = 1, # 반복되는 단어나 구절에 대한 패널티
    no_repeat_ngram_size: int = 0, ### 반복을 피하기 위해 고려할 n-gram 크기
    temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0
        ], ### 생성된 텍스트의 다양성을 제어한다. 0에 가까울수록 확신 있는 출력을 생성하고, 1에 가까울수록 더 다양한 출력을 생성함
    compression_ratio_threshold: Optional[float] = 2.4, # 압축 비율 임계값을 설정하여 너무 길거나 짧은 출력을 걸러낸다.
    log_prob_threshold: Optional[float] = -1.0, # 로그 확률 임계값을 설정하여 너무 낮은 확률의 출력을 걸러낸다.
    no_speech_threshold: Optional[float] = 0.6, # 말이 없는 것으로 간주되는 임계값을 설정함
    condition_on_previous_text: bool = True, # 이전 텍스트를 고려할지 여부를 결정함
    prompt_reset_on_temperature: float = 0.5, # 특정 온도에서 프롬푸트를 재설정할지 여부를 결정함
    initial_prompt: Optional[Union[str, Iterable[int]]] = None, # 초기 프롬프투 설정을 위한 문자열 또는 토큰 ID
    prefix: Optional[str] = None, # 모든 출력에 앞서 붙일 접두사 문자열
    suppress_blank: bool = True, # 빈 토큰을 억제할지 여부를 결정함
    suppress_tokens: Optional[List[int]] = [-1], # 제거할 토큰 ID 목록
    without_timestamps: bool = False, # 출력에 타임스탬프를 포함할지 여부를 결정함
    max_initial_timestamp: float = 1.0, # 초기 타임스탬프의 최대 값
    word_timestamps: bool = False, # 단어별 타임스탬프를 생성할지 여부를 결정함
    prepend_punctuations: str = "\"'“¿([{-", 
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    vad_filter: bool = False, # 음성 활동 탐지 필터를 사용할지 여부를 결정함
    vad_parameters: Optional[Union[dict, VadOptions]] = None, # VAD 필터의 매개변수
    max_new_tokens: Optional[int] = None, # 생성할 최대 토큰 수
    chunk_length: Optional[int] = None, ### 처리할 오디오 청크의 길이. 길이가 길수록 더 많은 메모리가 필요하지만, 더 정확한 결과를 얻을 수 있다.
    clip_timestamps: Union[str, List[float]] = "0", # 타임스탬프를 자를 위치
    hallucination_silence_threshold: Optional[float] = None, # 환각적인 소리를 제거하기 위한 임계값