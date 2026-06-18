# Humming Reels Search Project

### 허밍 기반 숏폼 밈/변형 음원 분석 시스템

허밍 또는 짧은 녹음 WAV를 업로드하면 저장된 원곡 데이터베이스를 기준으로 가장 유사한 구간을 찾고, 원곡 대비 Speed, Pitch, Chroma 변화를 분석하여 배속·피치 변형 및 숏폼 밈 스타일을 판별하는 AI 기반 음악 검색 프로젝트입니다.

본 프로젝트의 목표는 단순히 원곡 제목만 찾는 것이 아니라, 사용자가 릴스(Reels)와 쇼츠(Shorts)에서 들은 변형 음원이 원곡과 비교하여 어떻게 달라졌는지 설명하고, 관련 YouTube 후보 영상과 GPT 기반 분석 리포트를 제공하는 것입니다.

---

## Project Overview

기존 음악 검색 서비스는 원곡을 찾는 기능에 집중되어 있습니다.

본 프로젝트는 허밍 또는 짧은 녹음만으로 다음 정보를 함께 제공합니다.

* 원곡 정보
* 가장 유사한 원곡 구간
* Speed Factor 분석
* Pitch Shift 분석
* Chroma Similarity 분석
* 숏폼 밈 스타일 판별
* YouTube 후보 추천
* GPT 기반 분석 리포트

---

## Core Pipeline

```text
Original Song Database
        ↓
10-Second Segmentation
        ↓
Feature Extraction
(MFCC + Chroma + Pitch Contour)
        ↓
Similarity Search
        ↓
Speed/Pitch Refinement
        ↓
Transformation Analysis
        ↓
YouTube Candidate Ranking
        ↓
GPT Report Generation
```

---

## Tech Stack

### Language

* Python

### Audio Processing

* Librosa
* NumPy

### Machine Learning

* Scikit-learn

### Web Application

* Streamlit

### External APIs

* OpenAI API
* YouTube Data API

---

## Main Features

### 1. Audio Feature Extraction

다음 특징을 결합하여 음원 특성을 벡터화합니다.

* MFCC
* Chroma
* Pitch Contour

---

### 2. Segment-Level Search

원곡을 10초 단위 세그먼트로 분할한 뒤 입력 음원과 비교하여 가장 유사한 구간을 탐색합니다.

---

### 3. Transformation Detection

원곡 대비 다음 변화를 추정합니다.

* Speed Factor
* Pitch Shift
* Chroma Similarity
* Duration Ratio
* Match Confidence

---

### 4. YouTube Candidate Recommendation

YouTube Data API를 활용하여 다음 후보를 수집합니다.

* Official / MV
* Sped Up
* Remix
* Reels
* Shorts
* Trend / Meme

---

### 5. GPT-Based Report Generation

분석 결과를 기반으로 GPT 리포트를 생성합니다.

리포트에는 다음 정보가 포함됩니다.

* 원곡 정보
* 매칭 구간
* Speed 분석
* Pitch 분석
* Chroma 분석
* YouTube 후보
* 트렌드 해석

---

## Project Structure

```text
music_ai_project/
├── app/
│   ├── audio.py
│   ├── search.py
│   ├── segment.py
│   ├── song_catalog.py
│   ├── youtube.py
│   └── gpt.py
│
├── data/
│   └── README.md
│
├── report/
│   └── project_report.pdf
│
├── screenshots/
│
├── streamlit_app.py
├── test_segment.py
├── requirements.txt
└── README.md
```

---

## Example Result

```text
Original Song:
Justin Bieber - Beauty And A Beat

Matched Segment:
Beauty_original_start00060s.wav

Refined Match Score:
90.0%

Speed Factor:
1.25x

Pitch Shift:
+3 semitones

Chroma Similarity:
0.97

Detected Style:
Speed Up Shorts Version
```

---

## Dataset Notice

Original audio files are not included in this repository due to copyright considerations.

To run this project, place your own audio files inside:

```text
data/raw/songs/
data/raw/humming/
```

Generated files will automatically be created in:

```text
data/processed/segments/
data/processed/vectors/
```

---

## Current Limitations

* 긴 전체 음원보다 8~15초 정도의 짧은 허밍 입력에서 더 잘 동작합니다.
* YouTube 영상 오디오 직접 비교는 아직 구현되지 않았습니다.
* Speed/Pitch 정밀 매칭 과정은 정확도를 높이는 대신 실행 시간이 증가할 수 있습니다.

---

## Future Improvements

* YouTube 영상 오디오 직접 비교
* 더 빠른 벡터 인덱싱
* Pitch 추정 안정화
* GPT 리포트 품질 향상
* 숏폼 트렌드 분석 기능 강화

---

## Development Note

프로젝트 아이디어 선정, 시스템 설계, 알고리즘 구성, 실험 및 결과 분석은 직접 수행하였습니다.

구현 과정에서는 AI Coding Assistant(Codex)를 활용하여 코드 작성 및 디버깅을 보조받았습니다.

---

## Author

김지선

