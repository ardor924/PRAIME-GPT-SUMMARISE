# 새 PC용 구축 지침서 (SETUP.md)

> 이 문서는 **기존 원격 Git 저장소**만 가지고 **새로운 PC**에 동일한 개발/실행 환경을 빠르게 재현하기 위한 가이드입니다.  
> 대상 OS: Windows / macOS / Linux (명령어는 OS별 예시 제공)

---

## 1) 사전 요구사항

- Git
    
- Python **3.10.x** 권장 (3.9는 Pydantic v2 조합에서 경고/비호환 가능)
    
- (선택) Conda/Miniconda
    
- (선택) OpenAI API Key — 의미 기반 정규화(LLM) 사용 시
    

---

## 2) 저장소 클론

```bash
# 원하는 작업 디렉터리로 이동
cd ~/work   # Windows PowerShell: cd $HOME\work

# 원격 저장소 클론
git clone <YOUR_REMOTE_GIT_URL> farmlog
cd farmlog
```

> ⚠️ `.env`는 `.gitignore`에 의해 **원격에 존재하지 않습니다.** 5단계에서 새로 작성합니다.

---

## 3) Python 가상환경 만들기 (두 방법 중 택1)

### 방법 A) venv (권장, 단순)
```bash
# Python 3.10으로 venv 생성
python -m venv .venv

# 가상환경 활성화
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# pip 최신화
python -m pip install --upgrade pip

```



### 방법 B) Conda

```bash
# conda 환경 생성
conda create -n farmlog310 python=3.10 -y
conda activate farmlog310

# (선택) 추가로 프로젝트 로컬 venv를 만들고 싶다면
python -m venv .venv
# 이후엔 venv를 활성화해서 사용하거나, conda만 사용해도 됩니다.

```

> 💡 한 머신에서는 **venv만** 쓰는 걸 권장합니다. (conda+venv 동시 사용 시 혼동 주의)

---

## 4) Python 패키지 설치

저장소에는 `requirements.txt`와 `.requirements.sha256`(해시)이 포함되어 있습니다.  
프로젝트는 부트스트랩 시 이 해시를 사용해 요구사항 충족 여부를 자체 점검하지만, **첫 설치는 직접** 해주세요.

```
pip install -r requirements.txt
```

> ⚠️ 사내 프록시/미러가 있다면 `pip.ini`/`pip.conf` 등을 설정하세요.

---

## 5) .env 작성

원격에는 `.env`가 없으므로 로컬에서 새로 생성해야 합니다.  
아래 템플릿을 **프로젝트 루트**에 `.env`라는 이름으로 저장하세요.

```
# ─── 검색/KB/벡터스토어 ─────────────────────────────────
USE_WEB_SEARCH=1
RETRIEVE_TOP_K=4
CHROMA_DIR=./chroma
KB_DIR=./kb

# ─── 의미 게이트(임베딩) ───────────────────────────────
OPENAI_EMBED_MODEL=text-embedding-3-small
INTENT_POS_SIM=0.28
INTENT_MARGIN=0.06

# ─── 규칙 게이트 ───────────────────────────────────────
FARM_GATE_MIN_HITS=5
FARM_GATE_MIN_OP_HITS=1
FARM_GATE_BLOCK_NONFARM=true
NONFARM_BLOCK_MIN_HITS=2

# ─── STT CSV 경로 ──────────────────────────────────────
STT_CSV_DIR=./stt_csv
STT_CSV_FILENAME=qa.csv
CSV_GATE_LENIENT=1     # crop/operation 중 하나만 있어도 통과

# ─── 의미 정규화(LLM) 사용 ────────────────────────────
USE_SEMANTIC_NORMALIZER=1   # 0으로 두면 규칙 기반만 사용
OPENAI_MODEL=gpt-4o-mini    # LLM 사용 시
# OPENAI_API_KEY=sk-xxxxxxx  # LLM 사용시 필수. 없으면 규칙 기반만 동작

# ─── (선택) 검색 엔진 설정 ────────────────────────────
# Google CSE를 쓸 때만 필요. DuckDuckGo만 쓰면 비워두세요.
# GOOGLE_CSE_ID=xxxx
# GOOGLE_API_KEY=xxxx

```

> ✅ `.env`는 **절대 커밋하지 마세요.** (이미 `.gitignore`에 포함)

---

## 6) 프로젝트 폴더 구조 준비

최소한 다음 구조가 있으면 됩니다(없으면 생성됨).

```
farmlog/
├─ src/
│  ├─ app_fastapi.py
│  ├─ csv_io.py
│  ├─ gates.py
│  ├─ preprocess.py
│  ├─ semantic_normalize.py
│  └─ ... (기타 모듈, kb, rag, pipeline 등)
├─ kb/                # 지식 베이스 (없으면 생성/사용 안할 수도 있음)
├─ chroma/            # ChromaDB 저장 폴더 (없으면 실행중 생성)
├─ stt_csv/           # STT CSV 저장 루트 (직접 만듭니다)
│  └─ ID001/qa.csv    # 예: 하나 이상 예시 CSV
├─ text/              # 자유 텍스트 테스트용 (옵션)
├─ requirements.txt
├─ .requirements.sha256
└─ .env               # ← 방금 만든 파일
```

CSV 예시(UTF-8 또는 UTF-8-SIG 권장):

```
question,answer
재배지, 포장-2
작물, 배추
작업, 병해충관리
농약, 사파이어 입상수화제 1000배 200L
비료,
메모, 진딧물 소수 발견. 다음 주 예방 방제 예정

```

---

## 7) 서버 실행

`# 가상환경 활성화된 상태에서 python -m uvicorn src.app_fastapi:app --host 0.0.0.0 --port 8001`

콘솔에 아래가 보이면 정상:

```
# 가상환경 활성화된 상태에서
python -m uvicorn src.app_fastapi:app --host 0.0.0.0 --port 8001

```

---

## 8) 헬스체크 & 환경 확인

### 브라우저에서 확인

- [http://localhost:8001/healthz](http://localhost:8001/healthz)
    

예상 응답(일부 예시):

```
{
  "status": "ok",
  "vector_backend": "...",
  "stt_csv_dir": "C:/.../stt_csv",
  "stt_csv_filename": "qa.csv",
  "csv_gate_lenient": true,
  "use_semantic_normalizer": true
}

```

> `use_semantic_normalizer`가 **true**인데도 OpenAI 키가 없으면 LLM 단계는 건너뛰고 **규칙 기반 정규화만** 수행합니다.  
> LLM을 쓰고 싶다면 `.env`에 `OPENAI_API_KEY`를 넣고 서버를 재시작하세요.

---

## 9) 엔드포인트 빠른 테스트

### 9.1 CSV 요약 (경로 기반)

```bash
curl -X POST "http://localhost:8001/summarise_csv_json" ^
  -H "Content-Type: application/json" ^
  -d "{ \"path\": \"stt_csv/ID001/qa.csv\" }"
```

(macOS/Linux는 `^` 대신 `\` 사용)

정상일 경우 (예시):

```json
{
  "재배지": "포장-2",
  "작물": "배추",
  "작업": "병해충관리",
  "농약": "사파이어 입상수화제 1000배 200L",
  "비료": null,
  "메모": "배추 · 병해충관리 · 예방 방제 예정"
}

```

### 9.2 CSV 요약 (ID 기반)

```bash
curl -X POST "http://localhost:8001/summarise_csv_id" \
  -H "Content-Type: text/plain" \
  --data "ID001"
```

### 9.3 자유 텍스트 요약

```bash
curl -X POST "http://localhost:8001/summarise" \
  -H "Content-Type: application/json" \
  -d "{ \"stt_text\": \"포장-2 배추 진딧물 예찰, 다음 주 예방 방제 예정\" }"
```

---

## 10) Postman/REST Client 사용 팁

- `POST http://localhost:8001/summarise_csv_json`
    
- Body → raw → JSON:
    
    `{ "path": "stt_csv/ID001/qa.csv" }`
    
- 또는:
    
    `{ "id": "ID001" }`
    

---

## 11) 자주 있는 이슈 & 해결

- **`use_semantic_normalizer`가 false로 보임**
    
    - `.env`가 **서버 시작 전에** 로드되어야 합니다. 본 프로젝트는 모듈 임포트 시 `load_dotenv()`를 호출합니다.
        
    - `.env` 존재 여부/경로 확인 후 **서버 재시작**.
        
    - `USE_SEMANTIC_NORMALIZER=1`인지 확인. 오타/공백 제거.
        
- **LLM이 동작하지 않음**
    
    - `.env`에 `OPENAI_API_KEY`가 없으면 LLM은 건너뜁니다(규칙 기반만 수행). 키 추가 후 재시작.
        
    - 프록시/방화벽/회사망 이슈로 외부 통신이 막힌 경우 규칙 기반만 사용하세요.
        
- **`pydantic` V2 경고**
    
    - 코드상 V2 스키마로 맞춰져 있으므로 경고는 무해. 최신 버전으로 설치되어 있는지 `pip list | findstr pydantic` (Windows)로 확인.
        
- **CSV 파싱 오류 (`Could not determine delimiter`)**
    
    - CSV는 기본적으로 **콤마(,)** 구분을 기대. 값 내부 콤마가 있으면 `question,answer` 헤더 형태를 권장(첫 컬럼은 라벨, 나머지는 한 값으로 합침).
        
    - UTF-8-SIG 저장을 권장(엑셀로 저장 시 BOM 포함).
        
- **포트 충돌**
    
    - 이미 8001 사용 중이면 `--port 8002` 등으로 변경.
        

---

## 12) 운영/개발 워크플로우

- 변경 사항 작업 → 로컬에서 테스트 → Git 커밋/푸시
    
- 새 PC에서는 **3~5단계만** 최초 1회 수행 후,  
    이후엔 **`git pull` → (가상환경 활성화) → 서버 실행**만 하면 됩니다.
    
- `.env`는 PC별로 다를 수 있으니, 템플릿은 `/.env.example` 형태로 공유하세요.
    

---

## 13) 유지보수 팁

- **새 CSV가 들어와도** 규칙 기반 + (선택) LLM 정규화가 작동하도록 설계되어 있습니다.  
    키워드 추가가 필요하면 `src/preprocess.py`의 사전/정규식, 또는 `semantic_normalize.py`의 후처리 토큰을 보강하세요.
    
- RAG/KB 갱신이 필요하면:
    
```bash
curl -X POST "http://localhost:8001/ingest" \
  -H "Content-Type: application/json" \
  -d "{ \"kb_dir\": \"./kb\" }"
```
---
### 끝.