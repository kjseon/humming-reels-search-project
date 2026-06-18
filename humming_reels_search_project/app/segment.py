# segment.py - 오디오 세그멘테이션 처리
# wav 파일을 10초 단위, 5초 overlap으로 분할 저장합니다.

import os
import librosa
import soundfile as sf


def split_audio_segments(audio_path: str, output_folder: str) -> list[str]:
    """wav 파일을 10초 단위, 5초 overlap으로 분할하여 저장합니다.

    Args:
        audio_path:     입력 wav 파일 경로
        output_folder:  세그먼트 저장 디렉터리

    Returns:
        저장된 세그먼트 파일 경로 목록

    Example:
        >>> paths = split_audio_segments("data/raw/songs/track.wav", "data/processed/segments")
    """
    SEGMENT_DURATION = 10   # 세그먼트 길이 (초)
    OVERLAP_DURATION  = 0   # overlap 없음
    STEP_DURATION     = SEGMENT_DURATION - OVERLAP_DURATION  # 10초 간격

    y, sr = librosa.load(audio_path, sr=None, mono=True)  # 원본 sr 유지

    segment_samples = int(SEGMENT_DURATION * sr)
    step_samples    = int(STEP_DURATION * sr)

    os.makedirs(output_folder, exist_ok=True)

    base_name   = os.path.splitext(os.path.basename(audio_path))[0]
    saved_paths = []

    start_sample = 0
    while start_sample + segment_samples <= len(y):
        end_sample  = start_sample + segment_samples
        segment     = y[start_sample:end_sample]

        start_sec   = start_sample // sr          # 시작 시간 (초 단위, 파일명용)
        file_name   = f"{base_name}_start{start_sec:05d}s.wav"
        out_path    = os.path.join(output_folder, file_name)

        sf.write(out_path, segment, sr)
        saved_paths.append(out_path)
        print(f"[saved] {file_name}  ({start_sec}s ~ {start_sec + SEGMENT_DURATION}s)")

        start_sample += step_samples

    print(f"\n총 {len(saved_paths)}개 세그먼트 저장 완료 → {output_folder}")
    return saved_paths
