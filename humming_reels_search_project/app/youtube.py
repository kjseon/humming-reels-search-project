# youtube.py - YouTube Data API v3 기반 곡 검색

import os
import urllib.parse
import urllib.request
import json
import re
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def search_youtube(
    song_name: str,
    style_tag: str = "",
    max_results: int = 5,
) -> list[dict]:
    """곡명과 밈 스타일 태그로 YouTube 영상을 검색합니다.

    검색 쿼리 전략:
    - 배속 버전이면 "sped up" 키워드 추가
    - 숏폼 밈이면 "shorts meme" 키워드 추가
    - 기본: 곡명 + "official" or "lyrics"

    Args:
        song_name:   매칭된 곡 이름 (예: "Beauty reels remix")
        style_tag:   밈 스타일 태그 (예: "숏폼 밈 배속 버전")
        max_results: 반환할 최대 영상 수 (기본 5)

    Returns:
        [
            {
                "title":        영상 제목,
                "video_id":     유튜브 영상 ID,
                "url":          유튜브 링크,
                "thumbnail":    썸네일 URL,
                "channel":      채널명,
                "description":  설명 (첫 100자),
            },
            ...
        ]
    """
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY가 .env에 설정되지 않았습니다.")

    # 스타일에 따른 검색 키워드 구성
    keywords = []
    if "배속" in style_tag or "Speed Up" in style_tag:
        keywords.append("sped up")
    if "숏폼" in style_tag or "shorts" in style_tag.lower():
        keywords.append("shorts")
    if not keywords:
        keywords.append("official")

    query = f"{song_name} {' '.join(keywords)}"

    params = urllib.parse.urlencode({
        "part":       "snippet",
        "q":          query,
        "type":       "video",
        "maxResults": max_results,
        "key":        YOUTUBE_API_KEY,
        "relevanceLanguage": "ko",
        "safeSearch": "none",
    })

    req = urllib.request.Request(f"{SEARCH_URL}?{params}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for item in data.get("items", []):
        vid_id = item["id"].get("videoId", "")
        snippet = item.get("snippet", {})
        results.append({
            "title":       snippet.get("title", ""),
            "video_id":    vid_id,
            "url":         f"https://www.youtube.com/watch?v={vid_id}",
            "thumbnail":   snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            "channel":     snippet.get("channelTitle", ""),
            "description": snippet.get("description", "")[:100],
        })
    return results


def search_youtube_multi(
    song_name: str,
    style_tag: str = "",
    base_query: str | None = None,
) -> dict[str, list[dict]]:
    """공식 MV / 배속 밈 버전 / 쇼츠 밈 세 카테고리로 나눠 검색합니다.

    Args:
        song_name:  곡 이름
        style_tag:  밈 스타일 태그
        base_query: 실제 YouTube 검색에 사용할 곡명/아티스트 쿼리

    Returns:
        {
            "official": [...],   공식 MV / 원곡
            "sped_up":  [...],   배속 버전
            "shorts":   [...],   쇼츠/밈 영상
        }
    """
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY가 .env에 설정되지 않았습니다.")

    query_base = base_query or song_name

    def _search(extra_kw: str, order: str = "relevance", prefer_short: bool = False) -> list[dict]:
        query = f"{query_base} {extra_kw}"
        params = urllib.parse.urlencode({
            "part":       "snippet",
            "q":          query,
            "type":       "video",
            "maxResults": 12,
            "key":        YOUTUBE_API_KEY,
            "safeSearch": "none",
            "order": order,
            "videoDuration": "short" if prefer_short else "any",
        })
        req = urllib.request.Request(f"{SEARCH_URL}?{params}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = []
        for item in data.get("items", []):
            vid_id = item["id"].get("videoId", "")
            snippet = item.get("snippet", {})
            items.append({
                "title":     snippet.get("title", ""),
                "video_id":  vid_id,
                "url":       f"https://www.youtube.com/watch?v={vid_id}",
                "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
                "channel":   snippet.get("channelTitle", ""),
                "description": snippet.get("description", "")[:160],
                "published_at": snippet.get("publishedAt", ""),
            })
        return _rank_videos(_attach_video_stats(items), prefer_short=prefer_short)[:5]

    return {
        "official": _search("official music video", order="relevance"),
        "sped_up": _merge_results([
            _search("sped up shorts", order="relevance", prefer_short=True),
            _search("speed up remix reels", order="relevance", prefer_short=True),
            _search("sped up trend", order="viewCount", prefer_short=True),
        ]),
        "shorts": _merge_results([
            _search("shorts meme", order="relevance", prefer_short=True),
            _search("reels trend", order="relevance", prefer_short=True),
            _search("shorts trend", order="date", prefer_short=True),
        ]),
    }


def _attach_video_stats(items: list[dict]) -> list[dict]:
    video_ids = [item["video_id"] for item in items if item.get("video_id")]
    if not video_ids:
        return items

    params = urllib.parse.urlencode({
        "part": "statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY,
    })
    req = urllib.request.Request(f"{VIDEOS_URL}?{params}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    stats_by_id = {}
    for item in data.get("items", []):
        stats = item.get("statistics", {})
        stats_by_id[item.get("id", "")] = {
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "duration": item.get("contentDetails", {}).get("duration", ""),
            "duration_seconds": _parse_iso8601_duration(
                item.get("contentDetails", {}).get("duration", "")
            ),
        }

    enriched = []
    for item in items:
        enriched.append({**item, **stats_by_id.get(item.get("video_id", ""), {})})
    return enriched


def _parse_iso8601_duration(duration: str) -> int | None:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return None
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _shorts_score(item: dict) -> int:
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    score = 0
    for keyword in ("shorts", "#shorts", "reels", "sped up", "speed up", "meme", "trend", "tiktok"):
        if keyword in text:
            score += 1
    duration_seconds = item.get("duration_seconds")
    if duration_seconds is not None and duration_seconds <= 75:
        score += 3
    return score


def _rank_videos(items: list[dict], prefer_short: bool = False) -> list[dict]:
    filtered = items
    if prefer_short:
        short_candidates = [
            item for item in items
            if _shorts_score(item) >= 2
        ]
        if short_candidates:
            filtered = short_candidates

    return sorted(
        filtered,
        key=lambda item: (
            _shorts_score(item) if prefer_short else 0,
            item.get("view_count", 0),
            item.get("published_at", ""),
        ),
        reverse=True,
    )


def _merge_results(groups: list[list[dict]]) -> list[dict]:
    merged = {}
    for group in groups:
        for item in group:
            video_id = item.get("video_id")
            if video_id and video_id not in merged:
                merged[video_id] = item
    return sorted(
        merged.values(),
        key=lambda item: (_shorts_score(item), item.get("view_count", 0), item.get("published_at", "")),
        reverse=True,
    )[:5]


def rank_youtube_matches(
    youtube_results: dict[str, list[dict]] | None,
    style_tag: str,
    speed_factor: float | None = None,
    pitch_shift_semitones: int | None = None,
    limit: int = 5,
) -> list[dict]:
    """허밍 분석 결과와 YouTube 후보 메타데이터를 비교해 가까운 후보를 랭킹합니다.

    YouTube Data API는 영상 오디오를 제공하지 않으므로 이 점수는 실제 오디오
    유사도가 아니라 제목/설명/길이/조회수/게시일 기반의 후보 점수입니다.
    """
    if not youtube_results:
        return []

    candidates = []
    for category, items in youtube_results.items():
        for item in items:
            scored = dict(item)
            scored["category"] = category
            scored["match_score"] = _metadata_match_score(
                item,
                category,
                style_tag,
                speed_factor,
                pitch_shift_semitones,
            )
            scored["match_reason"] = _metadata_match_reason(
                item,
                category,
                style_tag,
                speed_factor,
                pitch_shift_semitones,
            )
            candidates.append(scored)

    unique = {}
    for item in candidates:
        video_id = item.get("video_id")
        if not video_id:
            continue
        if video_id not in unique or item["match_score"] > unique[video_id]["match_score"]:
            unique[video_id] = item

    return sorted(
        unique.values(),
        key=lambda item: (item["match_score"], item.get("view_count", 0), item.get("published_at", "")),
        reverse=True,
    )[:limit]


def _metadata_match_score(
    item: dict,
    category: str,
    style_tag: str,
    speed_factor: float | None,
    pitch_shift_semitones: int | None,
) -> int:
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    style_lower = style_tag.lower()
    score = 0

    if category in ("sped_up", "shorts"):
        score += 2
    if "배속" in style_tag or "speed" in style_lower or (speed_factor and speed_factor >= 1.12):
        for keyword in ("sped up", "speed up", "nightcore", "fast", "remix"):
            if keyword in text:
                score += 4
    if speed_factor and speed_factor <= 0.89:
        for keyword in ("slowed", "reverb", "slow"):
            if keyword in text:
                score += 4
    if pitch_shift_semitones and abs(pitch_shift_semitones) >= 2:
        for keyword in ("pitch", "nightcore", "remix", "edit"):
            if keyword in text:
                score += 2
    score += _shorts_score(item)

    duration_seconds = item.get("duration_seconds")
    if duration_seconds is not None:
        if duration_seconds <= 75:
            score += 4
        elif duration_seconds <= 180:
            score += 1

    views = item.get("view_count", 0)
    if views >= 1_000_000:
        score += 3
    elif views >= 100_000:
        score += 2
    elif views >= 10_000:
        score += 1

    return score


def _metadata_match_reason(
    item: dict,
    category: str,
    style_tag: str,
    speed_factor: float | None,
    pitch_shift_semitones: int | None,
) -> str:
    reasons = []
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    if category == "sped_up":
        reasons.append("배속/리믹스 검색 후보")
    if category == "shorts":
        reasons.append("Shorts/Reels 검색 후보")
    if speed_factor and speed_factor >= 1.12 and any(k in text for k in ("sped up", "speed up", "nightcore")):
        reasons.append(f"입력 분석 speed {speed_factor:.2f}x와 배속 키워드가 일치")
    if pitch_shift_semitones and abs(pitch_shift_semitones) >= 2 and any(k in text for k in ("pitch", "nightcore", "remix", "edit")):
        reasons.append(f"입력 분석 pitch {pitch_shift_semitones:+d}st와 변형 키워드가 일치")
    duration_seconds = item.get("duration_seconds")
    if duration_seconds is not None and duration_seconds <= 75:
        reasons.append("짧은 영상 길이")
    if item.get("view_count"):
        reasons.append("조회수 기반 우선순위")
    return ", ".join(reasons) if reasons else "곡명 기반 YouTube 후보"
