# LLM 학습 정리

LangChain 기반으로 LLM을 다루며 학습한 내용을 정리한 저장소이다.
ChatOpenAI 기본 사용법부터 프롬프트 캐싱, 멀티모달, LCEL, 이미지/영상 생성, RAG,
그리고 이 개념들을 종합해 만든 화상채팅 요약 RAG 프로젝트까지 다룬다.

실제 코드는 각 노트북 파일(`.ipynb`)에 그대로 담겨 있으며, 이 문서는 노트북에 흩어져 있는
**개념적인 내용만** 모아 정리한 것이다.

---

## 노트북 개요

| 파일 | 주제 | 핵심 키워드 |
| :--- | :--- | :--- |
| `a_LLM.ipynb` | LLM 기초와 프롬프트 캐싱 | ChatOpenAI, temperature, LogProb, 스트리밍, 캐싱(InMemory/Redis/Semantic) |
| `b_multimodal.ipynb` | 멀티모달 모델 | CLIP, 공동 임베딩 공간, 대조 학습, 벡터 유사도 |
| `c_LCEL.ipynb` | LangChain Expression Language | Chain, 프롬프트 템플릿, OutputParser, Batch/Async/Parallel, Runnable |
| `d_image_generator.ipynb` | 이미지/영상 생성 | gpt-image, sora-2 |
| `e_RAG.ipynb` | 검색 증강 생성 | RAG 8단계, 임베딩, Vector DB, RAG-Anything |
| `f-Google-Generative-AI.ipynb` | Google 생성 모델 | Gemini, ChatGoogleGenerativeAI |

---

## 1. LLM 기초와 프롬프트 캐싱 (`a_LLM.ipynb`)

### ChatOpenAI
- OpenAI에서 개발한 대화형 AI와 상호작용하기 위한 표준화된 인터페이스를 제공한다.
- 단순 텍스트 완성이 아닌 '대화(Chat)' 형식에 최적화된 모델을 호출할 때 사용한다.
- `model_name`: 사용할 AI 모델의 구체적인 명칭을 작성한다.

### temperature
모델 답변의 창의성/일관성을 조절하는 값이다. 범위는 0.0 ~ 2.0이며 일반적으로 0~1 사이를 사용한다.

| 값 | 특징 | 적합한 작업 |
| :--- | :--- | :--- |
| 낮음 (예: 0.1) | 가장 확률이 높은 답변을 선택. 일관적이고 논리적 | 코딩, 사실 관계 확인 등 정확도가 중요한 작업 |
| 높음 (예: 0.9) | 다양한 단어를 선택. 창의적이고 의외성 있는 답변 | 소설 쓰기, 아이디어 발상 |

### LogProb (로그 확률)
- 토큰이란 문장을 구성하는 개별 단어나 문자 등의 요소를 의미하고, 확률은 모델이 그 토큰을 예측할 확률을 나타낸다.
- 특정 단어가 올 확률을 0~1 사이의 값으로 계산하는데, 문장이 길어질수록 곱해지는 확률이 작아진다.
  그래서 로그를 취해 확률 100%(1.0)는 `log(1.0) = 0`이 되고, 확률이 낮아질수록 -1, -5, -10처럼 음수로 표현된다.
- 값이 0에 가까울수록 AI가 그 단어를 선택할 확신이 높았다고 판단할 수 있다.
- 모델이 자기 답변에 얼마나 확신을 가졌는지, 어떤 부분에서 망설였는지 분석하는 기술적 지표이다.

### 스트리밍 출력 (invoke vs stream)

| 구분 | `invoke` (일괄 처리) | `stream` (스트리밍) |
| :--- | :--- | :--- |
| 작동 방식 | 답변이 100% 완성될 때까지 기다린 후 한꺼번에 반환 | 첫 단어가 생성되는 즉시 한 조각씩 실시간 전송 |
| 체감 속도 | 느림 (전체 답변 생성 시간만큼 대기) | 매우 빠름 (첫 글자가 나오는 즉시 출력 시작) |
| 적합한 상황 | 결과값을 받아 후처리해야 하는 내부 로직 | 챗봇 UI, 긴 글 생성 등 사용자 대기 시간이 중요한 서비스 |
| 로컬 캐싱 | 쉬움 (질문-답변 한 쌍을 그대로 매칭 가능) | 까다로움 (완료 전까지 조각 단위라 별도 캐싱 로직 필요) |

### 프롬프트 캐싱
반복하여 동일하게 입력으로 들어가는 토큰에 대한 비용을 아낄 수 있는 기능이다.
저장 위치(메모리/Redis)와 일치 방식(단어 일치/의미 일치)에 따라 여러 방식으로 나뉜다.

| 캐싱 방식 | 저장소 | 일치 방식 | 특징 |
| :--- | :--- | :--- | :--- |
| InMemoryCache | 로컬 RAM | 단어 일치 | dict 자료형과 유사. key가 정확히 동일해야 value를 가져옴. 유사도 분석 불가 |
| Dict Stream Caching | 로컬 RAM | 단어 일치 | stream은 조각으로 저장되므로, 다 합친 후 직접 저장해야 함 |
| RedisCache | Redis | 단어 일치 | key를 먼저 조회하고 없으면 API 호출 후 Redis에 캐싱 |
| Redis Stream Caching | Redis | 단어 일치 | Redis 조회 후 없으면 API 호출, 결과를 Redis에 캐싱 |
| RedisSemanticCache | Redis (벡터) | 의미 일치 | 질문 의도를 임베딩 벡터로 비교. RediSearch가 탑재된 redis-stack 필요 |
| Stream RedisSemanticCache | Redis (벡터) | 의미 일치 | 위 방식을 스트리밍 출력에 적용 |

#### RedisSemanticCache 동작 흐름
일반 RedisCache가 글자 하나하나가 똑같아야 하는 단어 일치 방식이라면,
RedisSemanticCache는 질문의 의도를 파악하는 의미 일치 방식이다.
변환된 임베딩 벡터를 저장하고 검색하려면 벡터 검색 엔진인 RediSearch 모듈이 탑재된 redis-stack이 필요하다.

1. 질문: 사용자가 "폴란드의 수도는 어디야?"라고 질문한다.
2. 임베딩: 임베딩 모델이 질문을 벡터로 변환한다.
3. 검색: Redis에 저장된 기존 질문 벡터들과 거리를 비교한다.
4. 판단
   - 가장 가까운 데이터와의 거리가 설정한 `score_threshold`(예: 0.2) 이내라면 캐시 히트
   - 거리 안에 데이터가 없으면 API 호출 및 결과 저장
5. 반환: 캐시 히트 시 LLM 호출 없이 Redis에 저장된 답변을 즉시 반환한다.

#### redis-stack 실행 (Docker) 참고
- `docker run -d --name redis-stack -p 6380:6379 -p 8001:8001 redis/redis-stack-server:latest`
  - `-d` (detach): 실행만 시키고 백그라운드로 돌릴 때
  - `-p 접속포트:내부포트`: 서비스 포트와 GUI 포트를 각각 매핑
- `docker exec -it redis-stack redis-cli`
  - `-i` (interactive): 입력 스트림을 열어 키보드 명령을 컨테이너가 받게 함
  - `-t` (tty): 가상 터미널을 할당해 리눅스 터미널처럼 보이게 함

---

## 2. 멀티모달 모델 (`b_multimodal.ipynb`)

기존 LLM이 텍스트라는 단일 모드만 처리했다면, 멀티모달 모델은 여러 형태의 정보를 함께 처리할 수 있다.

| 개념 | 설명 |
| :--- | :--- |
| CLIP (Contrastive Language-Image Pre-training) | OpenAI가 발표한 모델로, 텍스트와 이미지를 하나의 공간에서 연결한다. 정해진 클래스가 아니라 자연어 설명을 통해 이미지의 맥락을 학습한다 |
| 공동 임베딩 공간 (Shared Embedding Space) | 텍스트와 이미지를 각각 특징 벡터로 변환한 뒤 하나의 좌표 평면에 배치한다. 텍스트 인코더는 문장을, 이미지 인코더는 사진을 벡터로 변환한다 |
| 대조 학습 (Contrastive Learning) | 관련 있는 텍스트-이미지 벡터는 가까워지도록, 관련 없는 데이터끼리는 멀어지도록 학습한다 |
| 벡터 유사도 (Vector Similarity) | 두 데이터가 얼마나 비슷한지 수학적으로 계산한다. 주로 코사인 유사도를 사용한다 |

벡터 유사도 해석
- 유사도가 1에 가까움: 이 텍스트는 이미지의 정확한 설명이다.
- 유사도가 0에 가까움: 이 텍스트와 이미지는 아무 상관이 없다.

---

## 3. LCEL — LangChain Expression Language (`c_LCEL.ipynb`)

- LangChain 프로토콜을 사용하여 복잡한 체인(Chain)을 더 쉽고 직관적으로 구축하기 위해 설계된 선언적 언어이다.
- 여러 구성 요소(프롬프트, 모델, 출력 파싱 등)를 리눅스의 파이프 연산자(`|`)와 유사한 방식으로 연결하여 하나의 흐름을 만든다.

### 프롬프트 구성 요소

| 요소 | 설명 |
| :--- | :--- |
| Chain 생성 | 프롬프트, 모델, 파서 등을 파이프로 연결해 하나의 실행 흐름을 만든다 |
| load_prompt | 외부 파일(yaml 등)에 저장된 프롬프트를 불러와 사용한다 |
| ChatPromptTemplate | 이전 대화 내용을 기억시켜 대화가 잘 이어지게 돕는 템플릿. `system`(시스템 설정), `human`(사용자 입력), `ai`(AI 답변) 메시지로 구성된다 |

### OutputParser 비교

| 파서 | 반환 형태 | 특징 |
| :--- | :--- | :--- |
| OutputParser | 문자열 | 답변을 문자열로 리턴하므로 추가 작업 없이 바로 사용 가능 |
| JsonOutputParser | dict | JSON 형식 답변을 dict로 리턴하므로 추가 변환 없이 사용 가능 |
| PydanticOutputParser | 사용자 정의 모델 객체 | JsonOutputParser보다 진화한 형태. 사용자가 정의한 데이터 모델 규격에 맞춰 가져온다 |

### 실행 방식 비교

| 방식 | 설명 |
| :--- | :--- |
| Batch | 여러 개의 질문을 list로 받아 일괄 처리한다. 하나의 template에 여러 개의 데이터를 넣는다 |
| Async (비동기) | 단일 쓰레드 환경에서도 여러 요청을 효율적으로 처리하고 서버 응답성을 유지하는 핵심 기술. 한 요청 처리 중에도 다른 요청을 받을 수 있어 서버가 멈추지 않는다 |
| Parallel | 하나의 데이터를 여러 개의 template에 넣는다 (Batch와 반대 방향) |

### Runnable 구성 요소

| 요소 | 설명 |
| :--- | :--- |
| RunnablePassthrough | LCEL 파이프라인은 한 방향으로 흐르기 때문에 중간에 변수를 추가하기 어렵다. 앞의 정보가 사라지지 않게 보존하면서 새로운 정보를 추가하는 전용 통로 역할을 한다 |
| RunnableLambda | 외부 함수를 LangChain 체인 안에서 실행할 수 있도록 해준다 |

---

## 4. 이미지/영상 생성 (`d_image_generator.ipynb`)

### 이미지 생성 모델 비교

| 항목 | gpt-image-1.5 | gpt-image-1-mini |
| :--- | :--- | :--- |
| 품질 | 매우 높음 | 보통 |
| 속도 | 느린 편 | 매우 빠름 |
| 비용 | 상대적으로 비쌈 | 저렴 |
| 디테일 표현 | 복잡한 장면, 인물, 텍스처에 강함 | 단순한 스타일 위주 |
| 텍스트 반영 | 프롬프트를 잘 이해 | 일부 단순화될 수 있음 |
| 추천 용도 | 작품, 디자인, 마케팅 이미지 | 테스트, 썸네일, 대량 생성 |

### 영상 생성
- sora-2: AI 영상 제작 모델. 품질이 높은 만큼 비용 부담이 큰 편이다.

---

## 5. RAG — 검색 증강 생성 (`e_RAG.ipynb`)

RAG(Retrieval-Augmented Generation)는 생성형 AI가 학습하지 않은 최신 정보나
특정 조직의 내부 데이터를 바탕으로 답변할 수 있도록 돕는 기술이다.

| 단계 | 의미 | 설명 |
| :--- | :--- | :--- |
| R (Retrieval) | 검색 | 질문과 관련된 문서를 방대한 데이터베이스에서 검색한다 |
| A (Augmentation) | 증강 | 검색된 정보를 질문과 결합하여 프롬프트를 보강한다 |
| G (Generation) | 생성 | 보강된 맥락을 바탕으로 최종 답변을 생성한다 |

### RAG 개발 순서 (8단계)

| 순서 | 단계 |
| :--- | :--- |
| 1 | 문서 불러오기 |
| 2 | 문서 분할하기 |
| 3 | 임베딩 |
| 4 | Vector DB 생성 및 저장 |
| 5 | 검색기 생성 |
| 6 | 프롬프트 생성 |
| 7 | LLM 생성 |
| 8 | 체인 실행 |

### RAG-Anything
다양한 형식의 문서를 분석하여 그래프 기반의 멀티모달 지식 베이스를 구축하고,
이를 통해 정교한 질의응답을 수행하는 차세대 RAG 프레임워크이다.
([RAG-Anything GitHub](https://github.com/HKUDS/RAG-Anything/tree/main))

| 단계 | 구성 | 설명 |
| :--- | :--- | :--- |
| 1. Multi-modal Content Parsing | Hierarchical Text Extraction | 문서의 제목, 본문 등 계층적 구조를 유지하며 텍스트 추출 |
| | Image Caption & Metadata | 이미지 내 객체를 감지하고 캡션을 생성하며 메타데이터 추출 |
| | LaTeX Equation Recognition | 복잡한 수식을 기계가 읽을 수 있는 LaTeX 형식으로 변환 |
| | Table Structure Parsing | 표의 행과 열 관계를 유지하며 데이터 구조화 |
| 2. Graph-based Knowledge Grounding | 텍스트 정보 처리 | LLM으로 텍스트 내 주요 개념과 관계 파악 |
| | 지식 그래프 생성 | 추출된 정보를 노드와 엣지로 구성된 그래프로 저장 |
| | VLM/LLM 프로세서 | 비전 모델(VLM)로 텍스트와 이미지/표 사이의 논리적 연결 생성 |
| | Merged KG | 여러 문서의 개별 그래프를 하나의 거대한 지식 그래프로 통합 |
| 3. Hybrid Query & Retrieval | Graph-based Retrieval | 지식 그래프를 탐색하여 질문과 관련된 개념적 관계망 추적 |
| | Embedding-based Retrieval | 벡터 DB로 질문과 유사한 텍스트 청크 및 이미지 정보 검색 |
| | 정보 통합 | 그래프 검색 결과와 벡터 검색 결과를 결합하여 LLM에 전달 |

참고: 표의 방향이 역방향인 경우 분석 능력이 감소할 수 있으며, 이미지보다 텍스트를 우선시하는 경향이 있다.

---

## 6. Google Generative AI (`f-Google-Generative-AI.ipynb`)

- Google AI의 `gemini`, `gemini-vision` 모델 및 다른 생성 모델에 접근하려면
  `langchain-google-genai` 통합 패키지의 `ChatGoogleGenerativeAI` 클래스를 사용한다.
- 멀티모달 작업에는 `gemini-1.5-pro` 모델을 사용한다.
- API KEY는 Google AI Studio에서 발급받아 환경 변수 `GOOGLE_API_KEY`로 설정한다.

---

## 실행 환경

- 필요한 라이브러리는 `requirements.txt`로 설치한다.
  - RAG-Anything 관련 의존성은 `requirements_rag_anything.txt`에 별도로 정리되어 있다.
- 루트 디렉토리에 `.env` 파일을 생성하고 API KEY 등 비밀정보를 입력한다.
  - 예: `OPENAI_API_KEY=sk-proj-...`
  - `.env`는 비밀정보이므로 저장소에 포함되지 않는다(`.gitignore` 처리).
