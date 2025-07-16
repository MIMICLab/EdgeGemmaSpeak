# Aria + AgentVox Integration

Project Aria 안경의 실시간 시선 추적과 AgentVox 음성 어시스턴트를 결합한 혁신적인 시선 기반 멀티모달 AI 어시스턴트입니다.

## 🌟 주요 기능

- **🎯 시선 기반 상호작용**: 말할 때마다 현재 보고 있는 것을 자동으로 분석
- **👁️ 실시간 시선 추적**: Project Aria 안경의 eye tracking으로 정확한 시선 위치 파악
- **🖼️ 자동 gaze 오버레이**: RGB 이미지에 시선 위치를 초록색 점으로 자동 표시
- **🤖 멀티모달 AI**: 음성 + 시각 + 시선 정보를 종합한 지능적 대화
- **📍 정밀 좌표 전달**: 이미지 내 시선 위치를 상대좌표(0.0-1.0)로 정확히 전달
- **🗣️ 한국어/영어 지원**: 다국어 음성 인식 및 합성

## ✨ 동작 원리

### 혁신적인 시선 기반 상호작용
1. **음성 입력 감지**: 사용자가 말을 시작하는 순간
2. **시선 위치 파악**: 현재 eye tracking으로 정확한 시선 방향 계산
3. **이미지 자동 캡처**: RGB 카메라에서 현재 뷰를 즉시 캡처
4. **gaze 포인트 오버레이**: 캡처된 이미지에 시선 위치를 초록색 점으로 표시
5. **좌표 정보 전달**: 시선 위치의 상대좌표를 텍스트와 함께 AI에 전달
6. **맞춤형 응답**: AI가 시선 위치를 중심으로 이미지를 분석하여 정확한 답변 제공

### 시선 정보 활용
- **"이것이 뭐야?"** → 시선이 향한 특정 객체 분석
- **"여기 뭐라고 쓰여있어?"** → 시선 위치의 텍스트 인식 및 읽기
- **"이 부분 설명해줘"** → 시선 영역 중심의 상세 설명

## 📋 요구사항

### 하드웨어
- Project Aria 스마트 안경 (WiFi 연결 가능)
- CUDA 지원 GPU (권장) 또는 충분한 CPU/RAM

### 소프트웨어
- Python 3.8+
- Project Aria SDK
- AgentVox (multimodal support)
- 필요한 Python 패키지들 (requirements.txt 참조)

### 모델 파일
- Gemma 3 12B 모델 (Q4_K_M 양자화)
- Multimodal projection 모델 (mmproj-gemma-3-12b-it-F16.gguf)
- Project Aria eye tracking 모델

## 🚀 설치 및 설정

### 1. 모델 다운로드

먼저 AgentVox의 멀티모달 모델들을 다운로드합니다:

```bash
# 기본 텍스트 모델만 다운로드
agentvox --download-model

# 멀티모달 모델까지 함께 다운로드 (권장)
agentvox --download-model --multimodal
```

### 2. Project Aria 안경 연결

WiFi를 통해 Project Aria 안경을 연결합니다:

```bash
# Aria 안경의 IP 주소 확인
# 안경의 설정에서 WiFi IP를 확인하세요
```

### 3. 실행

#### WiFi 연결 (IP 주소 필요)
```bash
python aria_agentvox_integration.py --device-ip <ARIA_IP_ADDRESS>
```

예시:
```bash
python aria_agentvox_integration.py --device-ip 192.168.0.23
```

#### USB 연결 (IP 주소 불필요)
```bash
python aria_agentvox_integration.py --interface usb
```

## 🎮 사용법

### 기본 실행

```bash
# WiFi 연결 + 한국어 (기본값)
python aria_agentvox_integration.py --device-ip 192.168.0.23

# USB 연결 + 한국어 (기본값)
python aria_agentvox_integration.py --interface usb

# 영어로 실행 (WiFi)
python aria_agentvox_integration.py --device-ip 192.168.0.23 --language en

# 영어로 실행 (USB)  
python aria_agentvox_integration.py --interface usb --language en
```

### 시선 기반 자동 상호작용

**🔥 핵심 특징**: 별도의 명령어 없이 자연스러운 대화만으로 시선 기반 분석이 자동으로 이루어집니다!

#### 자동 동작 방식
1. **말하기 시작** → 자동으로 현재 시선 위치의 이미지 캡처
2. **gaze 포인트 표시** → 초록색 점으로 시선 위치 명확히 표시
3. **좌표 전달** → 정확한 시선 위치를 AI에 전달
4. **맞춤형 분석** → 시선 중심의 정확한 답변 제공

#### 자연스러운 대화 예시

```
👁️ 사용자가 책의 특정 단어를 보면서...
🗣️ "이 단어가 무슨 뜻이야?"
🤖 어시스턴트: "시선이 향한 'paradigm'이라는 단어는 '패러다임'으로, 
   어떤 시대나 분야에서 지배적인 사고 체계나 관점을 의미합니다."

👁️ 사용자가 컴퓨터 화면의 에러 메시지를 보면서...
🗣️ "이 오류 어떻게 해결해?"
🤖 어시스턴트: "화면에 보이는 'Connection timeout' 오류는 네트워크 
   연결 문제입니다. 인터넷 연결을 확인하고 방화벽 설정을 점검해보세요."

👁️ 사용자가 요리책의 레시피를 보면서...
🗣️ "이 재료들로 뭘 만드는 거야?"
🤖 어시스턴트: "시선이 향한 재료 목록을 보니 파스타 알리오 올리오를 
   만드는 레시피네요. 마늘과 올리브오일이 핵심 재료입니다."
```

#### 특수 명령어 (시스템 제어용)
- **"상태"** - 시스템 상태 확인 (RGB/Eye/Gaze 상태)  
- **"이미지 클리어"** - 이미지 버퍼 초기화 (문제 해결용)
- **"종료"** - 프로그램 종료

### 지능적 시선 이해

시스템은 다음과 같은 방식으로 시선 정보를 활용합니다:

1. **객체 인식**: "이것이 뭐야?" → 시선이 향한 특정 물체 식별
2. **텍스트 읽기**: "여기 뭐라고 써있어?" → 시선 위치의 문자 인식
3. **상황 분석**: "이 상황은 어때?" → 시선 영역 중심의 상황 판단
4. **문제 해결**: "이거 어떻게 해?" → 시선이 향한 문제점 분석 및 해결책 제시

## ⚙️ 고급 설정

### 명령줄 옵션

```bash
python aria_agentvox_integration.py [OPTIONS]

Aria 연결 옵션:
  --interface {usb,wifi}  연결 방식 (기본값: wifi)
  --device-ip IP          Aria 기기 IP 주소 (WiFi 연결시 필수)
  --profile-name PROFILE  스트리밍 프로필 (기본값: profile18)

AI 모델 옵션:
  --llm-model PATH        LLM 모델 경로 (기본값: 자동 감지)
  --mmproj-model PATH     멀티모달 투영 모델 경로
  --device {cpu,cuda,mps,auto}  추론 장치 (기본값: auto)
  --language LANG         언어 설정 (기본값: ko)

통합 기능 옵션:
  --auto-capture          자동 이미지 캡처 활성화
  --capture-interval SEC  자동 캡처 간격 (기본값: 3.0초)
  --update-iptables       iptables 업데이트 (Linux만)

Eye Tracking 모델 옵션:
  --eye-model-path PATH   Eye tracking 모델 경로
  --eye-config-path PATH  Eye tracking 설정 경로
```

### 성능 최적화

1. **GPU 사용**: CUDA가 가능한 시스템에서는 `--device cuda` 사용
2. **메모리 관리**: 이미지 버퍼가 자동으로 관리되지만, 필요시 수동으로 클리어
3. **네트워크**: 안정적인 WiFi 연결 확보

## 🔧 문제 해결

### 일반적인 문제들

#### 1. Aria 연결 실패
```bash
# iptables 업데이트 (Linux)
python aria_agentvox_integration.py --device-ip <IP> --update-iptables

# 네트워크 연결 확인
ping <ARIA_IP_ADDRESS>
```

#### 2. 모델 로딩 실패
```bash
# 모델 재다운로드
agentvox --download-model --multimodal

# 모델 경로 확인
ls ~/.agentvox/models/
```

#### 3. 성능 문제
- GPU 메모리 부족: 더 작은 배치 크기 사용
- CPU 과부하: `--device cuda` 옵션으로 GPU 사용
- 네트워크 지연: 더 안정적인 WiFi 환경 확보

#### 4. 음성 인식 문제
- 마이크 권한 확인
- 배경 소음 최소화
- 명확한 발음으로 말하기

## 🎯 사용 예시

### 시나리오 1: 문서 읽기 도우미
```
1. 문서를 보면서 "캡처" 명령
2. "이 문서의 주요 내용을 요약해줘"
3. AI가 캡처된 이미지의 텍스트를 분석하여 요약 제공
```

### 시나리오 2: 실시간 학습 어시스턴트
```
1. 자동 캡처 모드 활성화: "자동 켜기"
2. 책이나 화면을 보면서 자유롭게 질문
3. "이 공식이 어떤 의미야?" 등의 질문
4. 시선 방향과 이미지를 기반으로 맞춤형 설명 제공
```

### 시나리오 3: 환경 인식 어시스턴트
```
1. 주변 환경을 보면서 "지금 뭐가 보여?"
2. 시선 방향과 RGB 이미지를 종합 분석
3. 객체 인식, 장면 설명, 안전 정보 등 제공
```

## 📚 API 참조

### AriaAgentVoxBridge 클래스

주요 메서드:
- `capture_current_view()`: 현재 뷰 수동 캡처
- `enable_auto_capture(interval)`: 자동 캡처 활성화
- `disable_auto_capture()`: 자동 캡처 비활성화
- `get_gaze_description()`: 현재 시선 방향 설명
- `get_status()`: 시스템 상태 반환

### VoiceAssistant 확장

새로운 메서드:
- `add_image(image)`: PIL 이미지 추가
- `add_image_from_path(path)`: 파일에서 이미지 추가
- `clear_images()`: 이미지 버퍼 클리어

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시다면:

1. Issue를 통해 버그 리포트나 기능 제안
2. Pull Request로 코드 개선사항 제출
3. 문서 개선 및 예시 추가

## 📄 라이선스

이 프로젝트는 Apache 2.0 라이선스 하에 제공됩니다.

## 🙏 감사의 말

- Meta의 Project Aria 팀
- Llama.cpp 개발진
- Gemma 모델 개발진
- 오픈소스 커뮤니티의 모든 기여자들