# audio.py - 허밍 기반 음악 검색용 특징 벡터 추출
# 구성: MFCC(13) + Chroma×2 가중치(24) + 피치 윤곽(20) = 57차원

import librosa
import numpy as np


def extract_mfcc(file_path: str, n_mfcc: int = 13, sr: int = 22050) -> np.ndarray:
    """MFCC 단독 벡터 (하위 호환용)."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    norm = np.linalg.norm(mfcc_mean)
    return mfcc_mean / (norm + 1e-9)


def _l2(v: np.ndarray) -> np.ndarray:
    """L2 단위 벡터 변환."""
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _extract_pitch_contour(y: np.ndarray, sr: int, n_bins: int = 20) -> np.ndarray:
    """piptrack으로 피치 에너지 분포를 추출해 히스토그램으로 정규화합니다.

    pyin보다 10배 이상 빠르며 멜로디 음역대 분포를 효과적으로 포착합니다.
    - pitches, magnitudes 행렬에서 에너지 강한 주파수만 선택
    - 로그 스케일 히스토그램 → L2 정규화

    Args:
        y:      오디오 신호
        sr:     샘플레이트
        n_bins: 피치 히스토그램 빈 수 (기본 20)

    Returns:
        shape (n_bins,) L2 정규화된 피치 분포 벡터
    """
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512, fmin=65.0, fmax=2093.0)

    # 각 프레임에서 가장 에너지 강한 주파수 선택
    mag_max_idx = np.argmax(magnitudes, axis=0)
    frame_pitches = pitches[mag_max_idx, np.arange(pitches.shape[1])]

    # 유효 피치만 (0 Hz 제외)
    valid = frame_pitches[frame_pitches > 0]
    if len(valid) == 0:
        return np.zeros(n_bins)

    # 로그 스케일 히스토그램
    log_p = np.log2(valid)
    hist, _ = np.histogram(log_p, bins=n_bins, range=(np.log2(65), np.log2(2093)))
    return _l2(hist.astype(float))


def extract_features_from_audio(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """이미 로드된 오디오 신호에서 결합 특징 벡터를 추출합니다."""
    if len(y) == 0:
        return np.zeros(n_mfcc + 12 + 20)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_vec = _l2(np.mean(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_vec = _l2(np.mean(chroma, axis=1)) * 2.0

    pitch_vec = _extract_pitch_contour(y, sr, n_bins=20) * 2.5

    combined = np.concatenate([mfcc_vec, chroma_vec, pitch_vec])
    return _l2(combined)


def extract_features(file_path: str, n_mfcc: int = 13, sr: int = 22050) -> np.ndarray:
    """MFCC + Chroma(×2 가중치) + 피치 윤곽 결합 특징 벡터를 추출합니다.

    허밍 매칭에 최적화된 3-요소 결합:
    ┌─────────────────────────────────────────────────────┐
    │  MFCC   (13차원, 가중치 1.0): 음색·배음 구조        │
    │  Chroma (12차원, 가중치 2.0): 멜로디 음계 분포      │  ← 가중치 2배
    │  Pitch  (20차원, 가중치 2.5): YIN 기반 실제 피치    │  ← 가중치 2.5배
    └─────────────────────────────────────────────────────┘
    각 요소를 L2 정규화 후 가중치 적용 → concat → 최종 L2 정규화

    Args:
        file_path:  입력 wav 파일 경로
        n_mfcc:     MFCC 계수 개수 (기본 13)
        sr:         리샘플링 샘플레이트 (기본 22050)

    Returns:
        shape (n_mfcc + 12 + 20,) = (45,) 결합 특징 벡터

    Example:
        >>> vec = extract_features("data/processed/segments/Beauty_reels_remix_start00000s.wav")
        >>> vec.shape
        (45,)
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return extract_features_from_audio(y, sr, n_mfcc=n_mfcc)
