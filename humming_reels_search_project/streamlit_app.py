# streamlit_app.py - 허밍 기반 릴스/쇼츠 밈 구간 검색 UI
# 실행: streamlit run streamlit_app.py

import os
import tempfile
import streamlit as st
from app.search import search_reel_segment, build_cache
from app.gpt import describe_song, build_fallback_report
from app.youtube import search_youtube_multi, rank_youtube_matches

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="릴스 구간 검색",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 허밍 기반 릴스/쇼츠 밈 구간 검색")
st.write("허밍 wav 파일을 업로드하면 가장 유사한 릴스 구간과 밈 맥락을 찾아드립니다.")

SEGMENTS_FOLDER = "data/processed/segments"
SONGS_FOLDER    = "data/raw/songs"

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    segments_folder = st.text_input(
        "세그먼트 폴더 경로", value=SEGMENTS_FOLDER,
        help="split_audio_segments로 생성된 세그먼트 폴더를 지정하세요."
    )
    wav_count = len([
        f for f in os.listdir(segments_folder) if f.lower().endswith(".wav")
    ]) if os.path.exists(segments_folder) else 0
    st.caption(f"📂 세그먼트 파일 수: {wav_count}개")

    st.divider()
    st.subheader("🗄️ 벡터 캐시")
    st.caption("처음 실행하거나 세그먼트가 추가됐을 때 눌러주세요.")
    if st.button("캐시 빌드 / 업데이트", use_container_width=True):
        if not os.path.exists(segments_folder) or wav_count == 0:
            st.error("세그먼트 폴더가 비어 있습니다.")
        else:
            with st.spinner("벡터 캐시 생성 중..."):
                new_count = build_cache(segments_folder)
            st.success("모든 캐시가 최신 상태입니다." if new_count == 0 else f"{new_count}개 파일 캐시 완료!")

    use_gpt     = st.toggle("💬 GPT SNS 트렌드 리포트 표시", value=True)
    use_youtube = st.toggle("▶ YouTube 관련 영상 표시", value=True)

# ── 파일 업로드 ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "허밍 wav 파일 업로드", type=["wav"],
    help="직접 허밍을 녹음한 wav 파일을 업로드하세요."
)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    if not os.path.exists(segments_folder):
        st.error(f"세그먼트 폴더를 찾을 수 없습니다: `{segments_folder}`")
        st.stop()
    if wav_count == 0:
        st.error("세그먼트 폴더에 wav 파일이 없습니다. `test_segment.py`를 먼저 실행해주세요.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── 검색 + 밈 판별 ───────────────────────────────────────
    with st.spinner("🔍 유사 구간 검색 및 밈 스타일 판별 중..."):
        try:
            result = search_reel_segment(
                tmp_path,
                segments_folder,
                SONGS_FOLDER,
            )
        except Exception as e:
            st.error(f"검색 중 오류가 발생했습니다: {e}")
            st.stop()
        finally:
            os.remove(tmp_path)

    ms = result["meme_style"]

    # ── 결과 출력 ─────────────────────────────────────────────
    st.divider()
    st.subheader("1. 원곡 정보")
    meta = result["song_metadata"]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"**{result['real_original_name']}**")
        if meta.get("artist"):
            st.caption(f"아티스트: {meta['artist']} · 피처링: {meta.get('featured_artist', 'N/A')}")
        st.caption(f"원곡 파일: {result['original_song_name']} · 총 길이 {result['db_duration']:.1f}초")
    with col2:
        st.metric("정밀 매칭", f"{result['transform_score']*100:.1f}%")
        st.caption(f"신뢰도: {result['match_confidence']}")

    st.divider()
    st.subheader("2. 매칭된 변형 음원")
    st.markdown(f"### {ms['style_tag']}")
    st.info(ms.get("summary", "변형 분석 요약을 생성하지 못했습니다."))
    st.caption(f"원곡 매칭 구간: {result['start_sec']}초 ~ {result['start_sec'] + 10}초")
    if os.path.exists(result["best_path"]):
        st.write("원곡에서 가장 가까운 구간")
        st.audio(result["best_path"], format="audio/wav")

    st.divider()
    st.subheader("3. 변형 분석")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        speed_factor = ms.get("speed_factor")
        st.metric("Speed Factor", f"{speed_factor:.2f}x" if speed_factor else "N/A")
    with col_b:
        st.metric("입력 BPM", f"{ms['bpm_humming']:.0f}" if ms.get("bpm_humming") else "N/A")
    with col_c:
        st.metric("원곡 구간 BPM", f"{ms['bpm_original_segment']:.0f}" if ms.get("bpm_original_segment") else "N/A")
    with col_d:
        pitch_shift = ms.get("pitch_shift_semitones")
        st.metric("Pitch Shift", f"{pitch_shift:+d} st" if pitch_shift is not None else "N/A")

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        st.metric("Chroma 유사도", f"{ms['chroma_similarity']:.2f}" if ms.get("chroma_similarity") is not None else "N/A")
    with col_f:
        st.metric("구간 길이 비율", f"{ms['duration_ratio']:.2f}" if ms.get("duration_ratio") else "N/A")
    with col_g:
        st.metric("원곡 커버리지", f"{ms['coverage_ratio']*100:.1f}%")

    with st.expander("분석 근거 자세히 보기"):
        st.write(f"- 1차 원형 매칭 점수: **{ms['base_match_score']*100:.1f}%**")
        st.write(f"- 변형 후보 정밀 매칭 점수: **{ms['transform_match_score']*100:.1f}%**")
        st.write(f"- 가장 가까운 변형 후보: **speed {ms['candidate_speed']:.2f}x / pitch {ms['candidate_pitch_shift']:+d} st**")
        st.write(f"- 허밍 길이: **{result['user_duration']:.1f}초**")
        st.write(f"- 원곡 매칭 구간 길이: **{ms['matched_segment_duration']:.1f}초**")
        st.write(f"- 원곡 전체 길이: **{ms['original_duration']:.1f}초**")
        st.write(f"- Raw 허밍 BPM: **{ms['bpm_humming_raw']:.0f}**")
        st.write(f"- 중앙 Pitch 비율: **{ms['pitch_ratio']:.2f}**" if ms.get("pitch_ratio") else "- 중앙 Pitch 비율: **N/A**")
        st.write(f"- 입력 Key: **{ms['input_key']}**, 원곡 구간 Key: **{ms['original_key']}**")

    # ── YouTube 관련 영상 ───────────────────────────────────
    youtube_results = None
    if use_youtube:
        st.divider()
        st.subheader("4. 실제 사용 영상")
        with st.spinner("YouTube 관련 영상 검색 중..."):
            try:
                youtube_results = search_youtube_multi(
                    song_name=result["matched_song_name"],
                    style_tag=ms["style_tag"],
                    base_query=result["song_metadata"]["youtube_query"],
                )
            except Exception as e:
                st.warning(f"YouTube 검색 실패: {e}")

        if youtube_results:
            best_youtube_matches = rank_youtube_matches(
                youtube_results,
                style_tag=ms["style_tag"],
                speed_factor=ms.get("speed_factor"),
                pitch_shift_semitones=ms.get("pitch_shift_semitones"),
                limit=3,
            )
            if best_youtube_matches:
                st.markdown("**허밍 분석과 가장 가까운 YouTube 후보**")
                st.caption("YouTube Data API 메타데이터 기반 후보입니다. 영상 오디오를 직접 비교한 결과는 아닙니다.")
                for idx, video in enumerate(best_youtube_matches, start=1):
                    title = video.get("title", "제목 없음")
                    channel = video.get("channel", "채널 정보 없음")
                    url = video.get("url", "")
                    views = video.get("view_count")
                    published = video.get("published_at", "")
                    duration = video.get("duration", "")
                    score = video.get("match_score", 0)
                    reason = video.get("match_reason", "")
                    view_text = f"조회수 {views:,}" if views is not None else "조회수 N/A"
                    st.markdown(f"{idx}. [{title}]({url}) · {channel}")
                    st.caption(f"후보 점수 {score} · {view_text} · {published} · {duration} · {reason}")

            category_labels = {
                "official": "공식/MV",
                "sped_up": "배속/리믹스",
                "shorts": "쇼츠/밈",
            }
            for category, videos in youtube_results.items():
                st.markdown(f"**{category_labels.get(category, category)}**")
                if not videos:
                    st.caption("검색 결과 없음")
                    continue
                for video in videos:
                    title = video.get("title", "제목 없음")
                    channel = video.get("channel", "채널 정보 없음")
                    url = video.get("url", "")
                    views = video.get("view_count")
                    published = video.get("published_at", "")
                    duration = video.get("duration", "")
                    view_text = f" · 조회수 {views:,}" if views is not None else ""
                    meta_text = f"{view_text} · {published} · {duration}"
                    st.markdown(f"- [{title}]({url}) · {channel}{meta_text}")

    # ── GPT SNS 트렌드 리포트 ────────────────────────────────
    st.divider()
    st.subheader("5. 트렌드 리포트")
    fallback_report = build_fallback_report(
        original_song_name=result["real_original_name"],
        style_tag=ms["style_tag"],
        start_sec=result["start_sec"],
        speed_factor=ms.get("speed_factor"),
        pitch_shift_semitones=ms.get("pitch_shift_semitones"),
        chroma_similarity=ms.get("chroma_similarity"),
        transform_score=result.get("transform_score"),
        youtube_results=youtube_results,
    )
    if use_gpt:
        with st.spinner("GPT 리포트 생성 중..."):
            try:
                gpt_text = describe_song(
                    song_name=result["matched_song_name"],
                    start_sec=result["start_sec"],
                    style_tag=ms["style_tag"],
                    matched_song_name=result["matched_song_name"],
                    original_song_name=result["real_original_name"],
                    song_metadata=result["song_metadata"],
                    coverage_ratio=ms["coverage_ratio"],
                    bpm_ratio=ms.get("bpm_ratio"),
                    version_bpm_ratio=ms.get("version_bpm_ratio"),
                    version_duration_ratio=ms.get("version_duration_ratio"),
                    speed_factor=ms.get("speed_factor"),
                    pitch_shift_semitones=ms.get("pitch_shift_semitones"),
                    pitch_ratio=ms.get("pitch_ratio"),
                    chroma_similarity=ms.get("chroma_similarity"),
                    similarity_score=result["score"],
                    user_duration=result["user_duration"],
                    db_duration=result["db_duration"],
                    bpm_humming=ms.get("bpm_humming"),
                    bpm_original=ms.get("bpm_original"),
                    bpm_matched=ms.get("bpm_matched"),
                    best_segment=result["best_segment"],
                    youtube_results=youtube_results,
                )
                if gpt_text and gpt_text.strip():
                    st.markdown(gpt_text)
                else:
                    st.warning("GPT 응답이 비어 있어 기본 리포트를 표시합니다.")
                    st.markdown(fallback_report)
            except Exception as e:
                st.warning(f"GPT 리포트 생성 실패: {e}")
                st.markdown(fallback_report)
    else:
        st.caption("사이드바의 GPT 리포트 토글이 꺼져 있어 기본 리포트를 표시합니다.")
        st.markdown(fallback_report)

    # ── 전체 유사도 ───────────────────────────────────────────
    with st.expander("📊 전체 세그먼트 유사도 보기"):
        for seg, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
            icon = "🟩" if seg == result["best_segment"] else "⬜"
            st.write(f"{icon} `{seg}` — **{score*100:.1f}%**")
            st.progress(float(max(0, score)))
