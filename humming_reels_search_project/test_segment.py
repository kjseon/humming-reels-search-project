# test_segment.py - split_audio_segments 함수 테스트
# 실행: python test_segment.py

import os
from app.segment import split_audio_segments

# 입력 파일 경로: 최종 구조에서는 raw/songs의 원곡만 세그먼트화합니다.
SONGS_FOLDER = "data/raw/songs"
INPUT_FILES = [
    os.path.join(SONGS_FOLDER, filename)
    for filename in sorted(os.listdir(SONGS_FOLDER))
    if filename.lower().endswith("_original.wav")
]

# 세그먼트 저장 경로
OUTPUT_FOLDER = "data/processed/segments"


def test_split(audio_path: str) -> None:
    """단일 파일에 대해 split_audio_segments를 실행하고 결과를 출력합니다."""
    print("=" * 60)
    print(f"[테스트 시작] {audio_path}")

    if not os.path.exists(audio_path):
        print(f"  [skip] 파일 없음, 건너뜀: {audio_path}\n")
        return

    saved = split_audio_segments(audio_path, OUTPUT_FOLDER)
    print(f"  [done] 완료: {len(saved)}개 세그먼트 저장\n")


if __name__ == "__main__":
    print("[start] split_audio_segments 테스트 시작\n")

    for file_path in INPUT_FILES:
        test_split(file_path)

    print("=" * 60)
    print("[done] 모든 테스트 완료")
    print(f"[output] 출력 폴더: {os.path.abspath(OUTPUT_FOLDER)}")
