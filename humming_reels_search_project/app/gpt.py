# gpt.py - OpenAI GPT API 연동 (학교 API Gateway 경유)

import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    http_client=httpx.Client(),
)

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
GPT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "8000"))


def describe_song(
    song_name: str,
    start_sec: int = 0,
    style_tag: str = "",
    matched_song_name: str = "",
    original_song_name: str = "",
    song_metadata: dict | None = None,
    coverage_ratio: float | None = None,
    bpm_ratio: float | None = None,
    version_bpm_ratio: float | None = None,
    version_duration_ratio: float | None = None,
    speed_factor: float | None = None,
    pitch_shift_semitones: int | None = None,
    pitch_ratio: float | None = None,
    chroma_similarity: float | None = None,
    similarity_score: float | None = None,
    user_duration: float | None = None,
    db_duration: float | None = None,
    bpm_humming: float | None = None,
    bpm_original: float | None = None,
    bpm_matched: float | None = None,
    best_segment: str = "",
    youtube_results: dict[str, list[dict]] | None = None,
) -> str:
    """GPT를 사용해 검색된 곡, 밈 스타일, SNS 트렌드 리포트를 생성합니다.

    Args:
        song_name:       곡 이름 (파일명에서 파싱된 값)
        start_sec:       매칭된 구간 시작 시간 (초)
        style_tag:       밈 스타일 태그 (예: "숏폼 밈 배속 버전")
        matched_song_name: 매칭된 버전명
        original_song_name: 비교 기준 원곡명
        song_metadata:   실제 곡 메타데이터
        coverage_ratio:  허밍 길이 / 원본 길이 비율
        bpm_ratio:       허밍 BPM / 원본 BPM 비율
        version_bpm_ratio: 매칭 버전 BPM / 원곡 BPM 비율
        version_duration_ratio: 매칭 버전 길이 / 원곡 길이 비율
        speed_factor:   원곡 매칭 구간 대비 입력 속도 비율
        pitch_shift_semitones: chroma 기반 추정 pitch shift
        pitch_ratio:    중앙 pitch 비율
        chroma_similarity: chroma 유사도
        similarity_score: 허밍과 매칭 구간의 코사인 유사도
        user_duration:   업로드한 허밍 길이
        db_duration:     매칭된 원본 음원 길이
        bpm_humming:     허밍 BPM
        bpm_original:    원본 BPM
        bpm_matched:     매칭 버전 BPM
        best_segment:    매칭된 세그먼트 파일명
        youtube_results: YouTube 검색 결과 묶음

    Returns:
        SNS 트렌드 리포트 텍스트
    """
    time_info  = f"{start_sec}초~{start_sec + 10}초 구간"
    style_info = f"\n- 판별된 밈 스타일: **{style_tag}**" if style_tag else ""
    ratio_info = ""
    if matched_song_name:
        ratio_info += f"\n- 매칭된 버전: {matched_song_name}"
    if original_song_name:
        ratio_info += f"\n- 비교 기준 원곡: {original_song_name}"
    if coverage_ratio is not None:
        ratio_info += f"\n- 커버리지 비율: {coverage_ratio*100:.1f}% (허밍 길이 / 원본 길이)"
    if bpm_ratio is not None:
        ratio_info += f"\n- BPM 비율: {bpm_ratio:.2f} (허밍 BPM / 원본 BPM)"
    if version_bpm_ratio is not None:
        ratio_info += f"\n- 버전 BPM 비율: {version_bpm_ratio:.2f} (매칭 버전 BPM / 원곡 BPM)"
    if version_duration_ratio is not None:
        ratio_info += f"\n- 버전 길이 비율: {version_duration_ratio:.2f} (매칭 버전 길이 / 원곡 길이)"
    if speed_factor is not None:
        ratio_info += f"\n- 추정 speed factor: {speed_factor:.2f}x (입력 허밍 BPM / 원곡 매칭 구간 BPM)"
    if pitch_shift_semitones is not None:
        ratio_info += f"\n- 추정 pitch shift: {pitch_shift_semitones:+d} semitones"
    if pitch_ratio is not None:
        ratio_info += f"\n- 중앙 pitch 비율: {pitch_ratio:.2f}"
    if chroma_similarity is not None:
        ratio_info += f"\n- chroma 유사도: {chroma_similarity:.2f}"
    if similarity_score is not None:
        ratio_info += f"\n- 허밍-DB 매칭 유사도: {similarity_score*100:.1f}%"
    if user_duration is not None:
        ratio_info += f"\n- 허밍 길이: {user_duration:.1f}초"
    if db_duration is not None:
        ratio_info += f"\n- 원본 음원 길이: {db_duration:.1f}초"
    if bpm_humming is not None:
        ratio_info += f"\n- 허밍 BPM: {bpm_humming:.0f}"
    if bpm_original is not None:
        ratio_info += f"\n- 원본 BPM: {bpm_original:.0f}"
    if bpm_matched is not None:
        ratio_info += f"\n- 매칭 버전 BPM: {bpm_matched:.0f}"
    if best_segment:
        ratio_info += f"\n- 매칭 세그먼트 파일: {best_segment}"

    youtube_info = _format_youtube_results(youtube_results)
    metadata_info = _format_song_metadata(song_metadata)

    prompt = f"""아래 근거만 사용해서 SNS 트렌드 리포트를 작성해줘.

[실제 곡 메타데이터]
{metadata_info}

[입력 정보]
- 곡명: '{song_name}'
- 매칭 구간: {time_info}{style_info}{ratio_info}

[YouTube 검색 기반 참고 자료]
{youtube_info}

[작성 원칙]
- 반드시 허밍 분석값(유사도, 매칭 구간, 커버리지, BPM)을 근거로 설명해줘.
- 매칭된 버전과 비교 기준 원곡을 구분해서 설명해줘.
- 리믹스/배속/편집 여부는 speed factor, pitch shift, chroma 유사도, 길이 비율을 원곡 매칭 구간과 비교해서 판단해줘.
- YouTube 정보는 위 검색 결과의 제목, 채널명, 조회수, 게시일, 길이, 링크를 근거로만 활용해줘.
- 리포트에는 실제 YouTube 링크를 반드시 포함해줘.
- 제공된 YouTube 검색 결과에 없는 사실(조회수, 댓글 반응, 실제 유행 여부, 가사 내용)은 단정하지 마.
- "실시간 유행"은 수집된 최신/조회수/Shorts 후보 데이터 기준이라고 한정해서 표현해줘.
- 단순 요약이 아니라, 왜 이 변형이 숏폼/밈에서 먹히는지 맥락을 분석해줘.
- 전체 답변은 1,600자 이내로 작성해줘.

[출력 형식]

## 1. 원곡과 매칭 근거
원곡, 매칭 구간, 유사도, speed factor, pitch shift, chroma 유사도를 근거 중심으로 요약해줘.

## 2. 밈 맥락 분석
판별된 스타일({style_tag if style_tag else '알 수 없음'})이 어떤 감정, 속도감, 장면 전환에 어울리는지 설명해줘.
배속이면 왜 에너지, 훅, 컷 전환에 유리한지 설명하고, 피치 변화가 있으면 분위기가 어떻게 달라지는지도 설명해줘.

## 3. YouTube 근거 링크
참고할 만한 링크 2~3개를 제목, 채널, 조회수, 링크와 함께 정리해줘.
각 링크가 공식/배속/쇼츠/밈 후보 중 어떤 근거인지 한 줄씩 설명해줘.

## 4. 활용 가이드
이 매칭 구간({time_info})을 숏폼에서 어떻게 쓰면 좋은지 3가지 제안해줘.
각 제안은 어울리는 영상 유형, 컷 편집 방식, 자막 또는 연출 포인트를 포함해줘."""

    messages = [
            {
                "role": "system",
                "content": (
                    "너는 음악과 SNS 트렌드 전문가야. "
                    "릴스, 쇼츠, 틱톡 밈 문화와 배속 트렌드에 대한 깊은 지식을 갖고 있어. "
                    "오디오 분석 수치와 실제 링크 근거를 분리해서 설명하고, "
                    "창작자가 바로 써먹을 수 있는 활용 가이드를 구체적으로 제공해."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    response = _create_chat_completion(messages, max_tokens=GPT_MAX_TOKENS)
    text = _extract_text(response)
    if text:
        return text

    retry_messages = [
        messages[0],
        {
            "role": "user",
            "content": (
                "다음 근거만 사용해서 500자 이내 리포트를 작성해줘.\n\n"
                f"[곡] {original_song_name or song_name}\n"
                f"[스타일] {style_tag}\n"
                f"[구간] {time_info}\n"
                f"[분석]{ratio_info}\n\n"
                f"[YouTube]\n{youtube_info}\n"
            ),
        },
    ]
    retry_response = _create_chat_completion(retry_messages, max_tokens=GPT_MAX_TOKENS)
    return _extract_text(retry_response)


def _create_chat_completion(messages: list[dict], max_tokens: int):
    return client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )


def _extract_text(response) -> str:
    if not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(getattr(item, "text", ""))
        content = "\n".join(part for part in parts if part)
    return content.strip()


def build_fallback_report(
    original_song_name: str,
    style_tag: str,
    start_sec: int,
    speed_factor: float | None = None,
    pitch_shift_semitones: int | None = None,
    chroma_similarity: float | None = None,
    transform_score: float | None = None,
    youtube_results: dict[str, list[dict]] | None = None,
) -> str:
    """GPT 호출이 꺼져 있거나 실패했을 때 표시할 근거 기반 리포트."""
    time_info = f"{start_sec}초~{start_sec + 10}초"
    lines = [
        "## 1. 곡 소개",
        f"- 비교 기준 원곡: **{original_song_name}**",
        f"- 입력 오디오는 원곡의 **{time_info} 구간**과 가장 가깝게 매칭되었습니다.",
        "",
        "## 2. 밈 스타일 분석",
        f"- 판별 결과: **{style_tag}**",
    ]
    if transform_score is not None:
        lines.append(f"- 변형 후보 정밀 매칭 점수: **{transform_score * 100:.1f}%**")
    if speed_factor is not None:
        lines.append(f"- 원곡 구간 대비 추정 속도: **{speed_factor:.2f}x**")
    if pitch_shift_semitones is not None:
        lines.append(f"- 추정 피치 변화: **{pitch_shift_semitones:+d} semitones**")
    if chroma_similarity is not None:
        lines.append(f"- Chroma 유사도: **{chroma_similarity:.2f}**")
    if speed_factor is not None and speed_factor >= 1.12:
        lines.append("- 밈 맥락: 배속감이 있어 빠른 컷 전환, 하이라이트 등장, 자신감 있는 포즈 전환에 잘 맞습니다.")
    elif speed_factor is not None and speed_factor <= 0.89:
        lines.append("- 밈 맥락: 느린 무드 편집, 회상 장면, 감성적인 전환 컷에 잘 맞습니다.")
    else:
        lines.append("- 밈 맥락: 원곡의 훅이나 하이라이트를 비교적 그대로 살리는 숏폼 사용에 적합합니다.")
    if pitch_shift_semitones:
        lines.append("- 피치 변화가 있어 원곡보다 더 가볍거나 몽환적인 인상을 줄 수 있습니다.")

    lines.extend(["", "## 3. 실제 YouTube 링크 기반 참고 자료"])
    link_lines = _format_fallback_links(youtube_results)
    lines.extend(link_lines if link_lines else ["- YouTube 검색 결과가 아직 없습니다."])

    lines.extend([
        "",
        "## 4. SNS 활용 가이드",
        "- 위 분석은 업로드 파일명이 아니라 입력 오디오와 원곡 세그먼트의 유사도, 속도, 피치, chroma 비교값을 기준으로 합니다.",
        "- 영상 유형: 패션/메이크업 전환, 여행 하이라이트, 댄스 포인트처럼 짧은 임팩트가 필요한 영상에 적합합니다.",
        "- 컷 편집: 비트가 치는 지점에 0.5~1초 단위 컷을 맞추고, 훅 직전에는 살짝 정지하거나 줌인하면 변형 음원의 에너지가 잘 살아납니다.",
        "- 자막/연출: 'before -> after', 'wait for it', 'POV' 같은 짧은 문구를 첫 1~2초에 배치해 시청 유지율을 노리는 구성이 좋습니다.",
    ])
    return "\n".join(lines)


def _format_fallback_links(youtube_results: dict[str, list[dict]] | None) -> list[str]:
    if not youtube_results:
        return []
    category_labels = {
        "official": "공식/MV",
        "sped_up": "배속/리믹스",
        "shorts": "쇼츠/밈",
    }
    lines = []
    for category, items in youtube_results.items():
        if not items:
            continue
        lines.append(f"**{category_labels.get(category, category)}**")
        for item in items[:3]:
            title = item.get("title", "제목 없음")
            channel = item.get("channel", "채널 정보 없음")
            url = item.get("url", "")
            views = item.get("view_count")
            published = item.get("published_at", "")
            duration = item.get("duration", "")
            view_text = f"조회수 {views:,}" if views is not None else "조회수 N/A"
            lines.append(f"- [{title}]({url}) · {channel} · {view_text} · {published} · {duration}")
    return lines


def _format_youtube_results(youtube_results: dict[str, list[dict]] | None) -> str:
    if not youtube_results:
        return "- YouTube 검색 결과가 제공되지 않았음"

    category_labels = {
        "official": "공식/MV",
        "sped_up": "배속/리믹스",
        "shorts": "쇼츠/밈",
    }
    lines = []
    for category, items in youtube_results.items():
        label = category_labels.get(category, category)
        if not items:
            lines.append(f"- {label}: 검색 결과 없음")
            continue
        lines.append(f"- {label}:")
        for item in items[:3]:
            title = item.get("title", "").strip()
            channel = item.get("channel", "").strip()
            url = item.get("url", "").strip()
            views = item.get("view_count")
            view_text = f" / 조회수 {views:,}" if views is not None else ""
            published = item.get("published_at", "")
            duration = item.get("duration", "")
            meta_text = f"{view_text} / 게시일 {published} / 길이 {duration}"
            if title or channel:
                lines.append(f"  - {title} / {channel}{meta_text} / {url}")
    return "\n".join(lines)


def _format_song_metadata(song_metadata: dict | None) -> str:
    if not song_metadata:
        return "- 저장된 실제 곡 메타데이터 없음"

    artist = song_metadata.get("artist", "")
    featured = song_metadata.get("featured_artist", "")
    full_title = song_metadata.get("full_title", "")
    query = song_metadata.get("youtube_query", "")

    lines = [
        f"- 아티스트: {artist}" if artist else "",
        f"- 곡명: {full_title}" if full_title else "",
        f"- 피처링: {featured}" if featured else "",
        f"- YouTube 검색 기준 쿼리: {query}" if query else "",
    ]
    return "\n".join(line for line in lines if line)
