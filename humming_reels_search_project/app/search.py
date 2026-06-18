# search.py - 허밍 기반 음악 검색 + 밈 스타일 판별 로직

import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

from app.audio import extract_features, extract_features_from_audio
from app.song_catalog import get_song_metadata

VECTOR_CACHE_DIR = "data/processed/vectors"

# 밈 스타일 판별 임계값
COVERAGE_THRESHOLD = 0.70   # 허밍 길이 / 원본 길이 비율
BPM_RATIO_THRESHOLD = 1.12  # BPM 배속 판별 기준
SHORTS_DURATION_SECONDS = 60.0
REFINE_TOP_K = 5
SPEED_CANDIDATES = [0.85, 1.0, 1.12, 1.25, 1.35]
PITCH_CANDIDATES = [-3, 0, 3]
KEY_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]


# ── 캐시 유틸 ────────────────────────────────────────────────

def _cache_path(seg_path: str) -> str:
    base = os.path.splitext(os.path.basename(seg_path))[0]
    return os.path.join(VECTOR_CACHE_DIR, f"{base}.npy")


def get_or_compute_vector(seg_path: str) -> np.ndarray:
    """캐시가 있으면 로드, 없으면 계산 후 저장합니다."""
    cache = _cache_path(seg_path)
    if os.path.exists(cache):
        return np.load(cache)
    os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
    vec = extract_features(seg_path)
    np.save(cache, vec)
    return vec


def build_cache(segments_folder: str) -> int:
    """segments_folder 내 모든 wav 파일의 벡터를 미리 캐시합니다."""
    wav_files = [f for f in os.listdir(segments_folder) if f.lower().endswith(".wav")]
    new_count = 0
    for wav_file in sorted(wav_files):
        seg_path = os.path.join(segments_folder, wav_file)
        if not os.path.exists(_cache_path(seg_path)):
            get_or_compute_vector(seg_path)
            new_count += 1
            print(f"[cache] {wav_file}")
    return new_count


# ── BPM 측정 ─────────────────────────────────────────────────

def _estimate_bpm(path: str) -> float:
    """librosa로 오디오의 BPM을 추정합니다."""
    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # tempo가 배열로 반환될 수 있으므로 스칼라로 변환
    return float(np.atleast_1d(tempo)[0])


def _align_bpm_to_reference(bpm: float | None, reference_bpm: float | None) -> tuple[float | None, float | None]:
    """하프타임/더블타임으로 잡힌 BPM을 기준 BPM에 맞춰 보정합니다."""
    if not bpm or not reference_bpm or reference_bpm <= 0:
        return bpm, None

    candidates = [bpm * 0.5, bpm, bpm * 2.0]
    aligned_bpm = min(candidates, key=lambda value: abs(np.log(value / reference_bpm)))
    return aligned_bpm, aligned_bpm / reference_bpm


def _audio_profile(path: str) -> dict:
    """변형 분석에 필요한 오디오 프로필을 계산합니다."""
    y, sr = librosa.load(path, sr=22050, mono=True)
    return _audio_profile_from_audio(y, sr)


def _audio_profile_from_audio(y: np.ndarray, sr: int) -> dict:
    """이미 로드된 오디오 신호에서 변형 분석 프로필을 계산합니다."""
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    key_index = int(np.argmax(chroma_norm))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=512, fmin=65.0, fmax=2093.0)
    max_idx = np.argmax(magnitudes, axis=0)
    frame_pitches = pitches[max_idx, np.arange(pitches.shape[1])]
    valid_pitches = frame_pitches[frame_pitches > 0]
    median_pitch = float(np.median(valid_pitches)) if len(valid_pitches) else None

    return {
        "duration": float(duration),
        "bpm": float(np.atleast_1d(tempo)[0]),
        "chroma": chroma_norm,
        "key_index": key_index,
        "median_pitch": median_pitch,
    }


def _confidence_label(score: float, transform_score: float | None = None) -> str:
    reference = transform_score if transform_score is not None else score
    if reference >= 0.92:
        return "높음"
    if reference >= 0.82:
        return "보통"
    return "낮음"


def _key_name(index: int | None) -> str:
    if index is None:
        return "N/A"
    return KEY_NAMES[index % 12]


def _apply_transform(y: np.ndarray, sr: int, speed: float, pitch_shift: int) -> np.ndarray:
    transformed = y
    if speed != 1.0:
        transformed = librosa.effects.time_stretch(transformed, rate=speed)
    if pitch_shift != 0:
        transformed = librosa.effects.pitch_shift(transformed, sr=sr, n_steps=pitch_shift)
    return transformed


def _refine_match_with_transforms(
    humming_vec: np.ndarray,
    candidate_scores: dict[str, float],
    segments_folder: str,
) -> dict:
    """상위 원곡 세그먼트를 speed/pitch 후보로 변형해 재매칭합니다."""
    best = None
    top_candidates = sorted(candidate_scores.items(), key=lambda item: -item[1])[:REFINE_TOP_K]
    for wav_file, base_score in top_candidates:
        seg_path = os.path.join(segments_folder, wav_file)
        y, sr = librosa.load(seg_path, sr=22050, mono=True)
        for speed in SPEED_CANDIDATES:
            for pitch_shift in PITCH_CANDIDATES:
                if speed == 1.0 and pitch_shift == 0:
                    transform_vec = get_or_compute_vector(seg_path)
                else:
                    transformed = _apply_transform(y, sr, speed, pitch_shift)
                    transform_vec = extract_features_from_audio(transformed, sr)
                score = float(cosine_similarity(
                    humming_vec.reshape(1, -1),
                    transform_vec.reshape(1, -1),
                )[0][0])
                if best is None or score > best["transform_score"]:
                    best = {
                        "segment": wav_file,
                        "path": seg_path,
                        "base_score": base_score,
                        "transform_score": score,
                        "candidate_speed": speed,
                        "candidate_pitch_shift": pitch_shift,
                    }
    return best or {
        "segment": max(candidate_scores, key=candidate_scores.get),
        "path": os.path.join(segments_folder, max(candidate_scores, key=candidate_scores.get)),
        "base_score": max(candidate_scores.values()),
        "transform_score": max(candidate_scores.values()),
        "candidate_speed": 1.0,
        "candidate_pitch_shift": 0,
    }


def _compare_chroma(query_chroma: np.ndarray, reference_chroma: np.ndarray) -> tuple[int, float]:
    """두 chroma 벡터 사이의 최적 반음 shift와 유사도를 반환합니다."""
    best_shift = 0
    best_score = -1.0
    for shift in range(12):
        shifted = np.roll(reference_chroma, shift)
        score = float(np.dot(query_chroma, shifted))
        if score > best_score:
            best_score = score
            best_shift = shift
    semitone_shift = best_shift if best_shift <= 6 else best_shift - 12
    return semitone_shift, best_score


def _is_catalog_original_segment(filename: str) -> bool:
    song_name, _ = _parse_segment_name(filename)
    metadata = get_song_metadata(song_name)
    original_stem = metadata.get("original_file_stem", "").lower()
    segment_stem = os.path.splitext(filename)[0].lower()
    return bool(original_stem and segment_stem.startswith(f"{original_stem.lower()}_start"))


def _variant_label(speed_factor: float | None, pitch_shift: int | None, user_duration: float) -> tuple[str, str]:
    """원곡 대비 변형률을 사람이 읽는 밈 스타일로 변환합니다."""
    is_short = user_duration <= SHORTS_DURATION_SECONDS
    if speed_factor is not None and speed_factor >= BPM_RATIO_THRESHOLD:
        return "원곡 기반 배속 밈 버전 (Speed Up Shorts)", "speed_up"
    if speed_factor is not None and speed_factor <= 1 / BPM_RATIO_THRESHOLD:
        return "원곡 기반 감속/무드 밈 버전 (Slowed Edit)", "slowed"
    if pitch_shift is not None and abs(pitch_shift) >= 2:
        return "원곡 기반 피치/무드 변형 밈 버전", "pitch_shift"
    if is_short:
        return "원곡 하이라이트 숏폼 사용 버전", "shortform"
    return "원곡 유사 버전", "original_like"


def _analysis_summary(analysis: dict) -> str:
    parts = []
    speed = analysis.get("speed_factor")
    if speed is not None:
        if speed >= BPM_RATIO_THRESHOLD:
            parts.append(f"원곡 구간보다 약 {speed:.2f}배 빠르게 들립니다.")
        elif speed <= 1 / BPM_RATIO_THRESHOLD:
            parts.append(f"원곡 구간보다 약 {speed:.2f}배 느리게 들립니다.")
        else:
            parts.append("속도는 원곡과 거의 비슷합니다.")
    pitch_shift = analysis.get("pitch_shift_semitones")
    if pitch_shift:
        direction = "높아진" if pitch_shift > 0 else "낮아진"
        parts.append(f"chroma 기준 피치가 {abs(pitch_shift)}반음 {direction} 후보입니다.")
    chroma = analysis.get("chroma_similarity")
    if chroma is not None:
        parts.append(f"멜로디/화성 유사도는 {chroma:.2f}입니다.")
    return " ".join(parts)


# ── 원본 음원 경로 탐색 ──────────────────────────────────────

def _song_stem_from_parsed_name(song_name_parsed: str) -> str:
    """버전 태그를 제거한 기본 곡 stem을 반환합니다.

    예: "Beauty reels remix" -> "Beauty"
    """
    parts = song_name_parsed.replace("_", " ").split()
    version_words = {
        "original", "reels", "reel", "remix", "moonlight",
        "sped", "speed", "up", "slowed", "nightcore",
    }
    base_parts = [part for part in parts if part.lower() not in version_words]
    return " ".join(base_parts) if base_parts else song_name_parsed


def _find_song_path(song_name_parsed: str, songs_folder: str = "data/raw/songs") -> str | None:
    """파싱된 곡 이름으로 원본 음원 파일을 찾습니다.

    예: "Beauty reels remix" → "Beauty_reels_remix.wav"
    """
    if not os.path.exists(songs_folder):
        return None
    target = song_name_parsed.replace(" ", "_").lower()
    for f in os.listdir(songs_folder):
        if f.lower().endswith(".wav") and os.path.splitext(f.lower())[0] == target:
            return os.path.join(songs_folder, f)
    return None


def _find_original_song_path(song_name_parsed: str, songs_folder: str = "data/raw/songs") -> str | None:
    """매칭된 버전명과 같은 기본 곡의 original 파일을 찾습니다."""
    if not os.path.exists(songs_folder):
        return None

    metadata = get_song_metadata(song_name_parsed)
    catalog_original = metadata.get("original_file_stem", "").lower()
    stem = _song_stem_from_parsed_name(song_name_parsed).replace(" ", "_").lower()
    preferred = f"{stem}_original"
    fallback = None

    for f in os.listdir(songs_folder):
        if not f.lower().endswith(".wav"):
            continue
        name = os.path.splitext(f.lower())[0]
        if catalog_original and name == catalog_original:
            return os.path.join(songs_folder, f)
        if name == preferred:
            return os.path.join(songs_folder, f)
        if name.startswith(f"{stem}_") and "original" in name:
            fallback = os.path.join(songs_folder, f)
    return fallback


# ── 밈 스타일 판별 ───────────────────────────────────────────

def classify_meme_style(
    user_duration: float,
    original_duration: float,
    humming_path: str,
    original_song_path: str | None,
    matched_song_path: str | None = None,
    matched_song_name: str = "",
) -> dict:
    """coverage_ratio와 BPM ratio를 기반으로 밈 스타일을 판별합니다.

    Args:
        user_duration:      허밍 파일 길이 (초)
        original_duration:  원본 음원 전체 길이 (초)
        humming_path:       허밍 wav 경로 (BPM 측정용)
        original_song_path: 원본 음원 wav 경로 (BPM 측정용, None이면 BPM 측정 생략)
        matched_song_path:  매칭된 버전 wav 경로
        matched_song_name:  매칭된 버전명

    Returns:
        {
            "style_tag":       밈 스타일 태그 (str),
            "coverage_ratio":  허밍 길이 / 원본 길이 (float),
            "bpm_humming":     허밍 BPM (float),
            "bpm_original":    원본 BPM (float | None),
            "bpm_ratio":       BPM 비율 (float | None),
            "bpm_matched":     매칭 버전 BPM (float | None),
            "version_bpm_ratio": 매칭 버전 BPM / 원본 BPM (float | None),
            "version_duration_ratio": 매칭 버전 길이 / 원본 길이 (float | None),
            "branch":          판별 분기 "shortform" | "highlight" (str),
        }
    """
    coverage_ratio = user_duration / original_duration if original_duration > 0 else 0.0

    # BPM 측정
    bpm_humming = _estimate_bpm(humming_path)
    bpm_original = _estimate_bpm(original_song_path) if original_song_path else None
    bpm_humming_aligned, bpm_ratio = _align_bpm_to_reference(bpm_humming, bpm_original)
    bpm_matched = _estimate_bpm(matched_song_path) if matched_song_path else None
    bpm_matched_aligned, version_bpm_ratio = _align_bpm_to_reference(bpm_matched, bpm_original)
    matched_duration = librosa.get_duration(path=matched_song_path) if matched_song_path else None
    version_duration_ratio = (
        matched_duration / original_duration
        if matched_duration is not None and original_duration > 0
        else None
    )

    matched_lower = matched_song_name.lower()
    is_remix_name = any(word in matched_lower for word in ("remix", "reels", "moonlight", "sped", "speed"))
    is_speed_changed = (
        version_bpm_ratio is not None
        and (version_bpm_ratio >= BPM_RATIO_THRESHOLD or version_bpm_ratio <= 1 / BPM_RATIO_THRESHOLD)
    )
    is_length_changed = (
        version_duration_ratio is not None
        and abs(version_duration_ratio - 1.0) >= 0.15
    )

    # 분기 판별
    if is_remix_name or is_speed_changed or is_length_changed:
        branch = "remix"
        if version_bpm_ratio is not None and version_bpm_ratio >= BPM_RATIO_THRESHOLD:
            style_tag = "원곡 기반 배속 리믹스 버전"
        elif version_bpm_ratio is not None and version_bpm_ratio <= 1 / BPM_RATIO_THRESHOLD:
            style_tag = "원곡 기반 감속/무드 리믹스 버전"
        else:
            style_tag = "원곡 기반 리믹스/편집 버전"
    elif coverage_ratio >= COVERAGE_THRESHOLD:
        branch = "shortform"
        if bpm_ratio is not None and bpm_ratio >= BPM_RATIO_THRESHOLD:
            style_tag = "숏폼 밈 배속 버전 (Speed Up Meme)"
        else:
            style_tag = "일반 숏폼 밈 버전"
    else:
        branch = "highlight"
        if bpm_ratio is not None and bpm_ratio >= BPM_RATIO_THRESHOLD:
            style_tag = "원곡 기반 배속 버전"
        else:
            style_tag = "원곡 버전"

    return {
        "style_tag":      style_tag,
        "coverage_ratio": coverage_ratio,
        "bpm_humming":    bpm_humming_aligned,
        "bpm_humming_raw": bpm_humming,
        "bpm_original":   bpm_original,
        "bpm_ratio":      bpm_ratio,
        "bpm_matched":    bpm_matched_aligned,
        "bpm_matched_raw": bpm_matched,
        "version_bpm_ratio": version_bpm_ratio,
        "version_duration_ratio": version_duration_ratio,
        "branch":         branch,
    }


# ── 메인 검색 함수 ───────────────────────────────────────────

def search_reel_segment(
    humming_path: str,
    segments_folder: str,
    songs_folder: str = "data/raw/songs",
) -> dict:
    """허밍과 가장 유사한 릴스 구간을 찾고 밈 스타일을 판별합니다.

    Args:
        humming_path:    허밍 wav 파일 경로
        segments_folder: segment wav 파일들이 있는 폴더
        songs_folder:    원본 음원 wav 파일들이 있는 폴더 (BPM/길이 측정용)

    Returns:
        {
            "best_segment":  가장 유사한 segment 파일명 (str),
            "best_path":     가장 유사한 segment 전체 경로 (str),
            "score":         cosine similarity 점수 (float),
            "song_name":     파싱한 곡 이름 (str),
            "start_sec":     매칭 구간 시작 시간 (int, 초),
            "all_scores":    {파일명: score} 전체 결과 딕셔너리,
            "user_duration": 허밍 길이 (float, 초),
            "db_duration":   원본 음원 총 길이 (float, 초),
            "meme_style":    classify_meme_style 결과 딕셔너리,
        }
    """
    if not os.path.exists(humming_path):
        raise FileNotFoundError(f"허밍 파일을 찾을 수 없습니다: {humming_path}")
    if not os.path.exists(segments_folder):
        raise FileNotFoundError(f"세그먼트 폴더를 찾을 수 없습니다: {segments_folder}")

    wav_files = sorted([
        f for f in os.listdir(segments_folder)
        if f.lower().endswith(".wav") and _is_catalog_original_segment(f)
    ])
    if not wav_files:
        raise ValueError(
            "원곡 세그먼트가 없습니다. raw/songs의 *_original.wav 파일을 기준으로 "
            "test_segment.py를 다시 실행해 주세요."
        )

    # 1. 허밍 길이 측정
    user_duration = librosa.get_duration(path=humming_path)

    # 2. 코사인 유사도 검색
    humming_vec = extract_features(humming_path)
    all_scores = {}
    for wav_file in wav_files:
        seg_path = os.path.join(segments_folder, wav_file)
        seg_vec  = get_or_compute_vector(seg_path)
        score    = float(cosine_similarity(
            humming_vec.reshape(1, -1),
            seg_vec.reshape(1, -1)
        )[0][0])
        all_scores[wav_file] = score

    refined_match = _refine_match_with_transforms(humming_vec, all_scores, segments_folder)
    best_segment = refined_match["segment"]
    best_score   = refined_match["base_score"]
    best_path    = refined_match["path"]
    transform_score = refined_match["transform_score"]
    song_name, start_sec = _parse_segment_name(best_segment)
    song_metadata = get_song_metadata(song_name)

    original_song_path = _find_original_song_path(song_name, songs_folder)
    if original_song_path:
        db_duration = librosa.get_duration(path=original_song_path)
    else:
        # 원본 파일 없으면 세그먼트 파일명 기반으로 총 길이 추정
        # (start_sec + 세그먼트 길이로 하한 추정)
        db_duration = start_sec + 10.0
    original_song_name = (
        os.path.splitext(os.path.basename(original_song_path))[0].replace("_", " ")
        if original_song_path
        else song_name
    )
    real_original_name = (
        f"{song_metadata['artist']} - {song_metadata['full_title']}"
        if song_metadata.get("artist")
        else original_song_name
    )

    # 4. 원곡 매칭 구간 대비 입력 허밍 변형 분석
    humming_profile = _audio_profile(humming_path)
    segment_profile = _audio_profile(best_path)
    bpm_humming, speed_factor = _align_bpm_to_reference(
        humming_profile["bpm"],
        segment_profile["bpm"],
    )
    pitch_shift, chroma_similarity = _compare_chroma(
        humming_profile["chroma"],
        segment_profile["chroma"],
    )
    pitch_ratio = (
        humming_profile["median_pitch"] / segment_profile["median_pitch"]
        if humming_profile["median_pitch"] and segment_profile["median_pitch"]
        else None
    )
    duration_ratio = (
        user_duration / segment_profile["duration"]
        if segment_profile["duration"] > 0
        else None
    )
    candidate_speed = refined_match["candidate_speed"]
    candidate_pitch_shift = refined_match["candidate_pitch_shift"]
    refine_gain = transform_score - best_score
    effective_speed_factor = (
        candidate_speed
        if refine_gain >= 0.05 and candidate_speed != 1.0
        else speed_factor
    )
    effective_pitch_shift = (
        candidate_pitch_shift
        if refine_gain >= 0.05 and candidate_pitch_shift != 0
        else pitch_shift
    )

    style_tag, branch = _variant_label(effective_speed_factor, effective_pitch_shift, user_duration)
    matched_variant_name = f"{real_original_name} - {style_tag}"

    variation_analysis = {
        "style_tag": style_tag,
        "branch": branch,
        "match_confidence": _confidence_label(best_score, transform_score),
        "transform_match_score": transform_score,
        "base_match_score": best_score,
        "candidate_speed": candidate_speed,
        "candidate_pitch_shift": candidate_pitch_shift,
        "bpm_speed_factor": speed_factor,
        "speed_factor": effective_speed_factor,
        "duration_ratio": duration_ratio,
        "chroma_pitch_shift_semitones": pitch_shift,
        "pitch_shift_semitones": effective_pitch_shift,
        "pitch_ratio": pitch_ratio,
        "chroma_similarity": chroma_similarity,
        "bpm_humming": bpm_humming,
        "bpm_humming_raw": humming_profile["bpm"],
        "bpm_original_segment": segment_profile["bpm"],
        "bpm_original": segment_profile["bpm"],
        "bpm_ratio": speed_factor,
        "bpm_matched": bpm_humming,
        "version_bpm_ratio": speed_factor,
        "version_duration_ratio": duration_ratio,
        "coverage_ratio": user_duration / db_duration if db_duration > 0 else 0.0,
        "user_duration": user_duration,
        "matched_segment_duration": segment_profile["duration"],
        "original_duration": db_duration,
        "input_key_index": humming_profile["key_index"],
        "original_key_index": segment_profile["key_index"],
        "input_key": _key_name(humming_profile["key_index"]),
        "original_key": _key_name(segment_profile["key_index"]),
    }
    variation_analysis["summary"] = _analysis_summary(variation_analysis)

    return {
        "best_segment":  best_segment,
        "best_path":     best_path,
        "score":         best_score,
        "transform_score": transform_score,
        "match_confidence": variation_analysis["match_confidence"],
        "song_name":     song_name,
        "matched_song_name": matched_variant_name,
        "original_song_name": original_song_name,
        "real_original_name": real_original_name,
        "song_metadata": song_metadata,
        "matched_song_path": None,
        "original_song_path": original_song_path,
        "start_sec":     start_sec,
        "all_scores":    all_scores,
        "user_duration": user_duration,
        "db_duration":   db_duration,
        "meme_style":    variation_analysis,
        "variation_analysis": variation_analysis,
    }


def _parse_segment_name(filename: str) -> tuple[str, int]:
    """세그먼트 파일명에서 곡 이름과 시작 시간을 파싱합니다."""
    name = os.path.splitext(filename)[0]
    if "_start" in name:
        song_part, time_part = name.rsplit("_start", 1)
        song_name = song_part.replace("_", " ")
        try:
            start_sec = int(time_part.replace("s", ""))
        except ValueError:
            start_sec = 0
    else:
        song_name = name.replace("_", " ")
        start_sec = 0
    return song_name, start_sec


def find_best_match(query_features: np.ndarray, db: dict) -> str:
    """쿼리 특징 벡터와 가장 유사한 음악을 반환합니다. (하위 호환용)"""
    if not db:
        return "데이터베이스가 비어 있습니다."
    return max(
        db,
        key=lambda name: cosine_similarity(
            query_features.reshape(1, -1),
            db[name].reshape(1, -1)
        )[0][0]
    )
