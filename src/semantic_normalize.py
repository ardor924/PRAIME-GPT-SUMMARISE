# -*- coding: utf-8 -*-
"""
의미 기반 CSV 정규화기 v2.2
- LLM 사용 가능 시: few-shot으로 1차 정규화
- LLM 불가/실패 시: 규칙 기반 오프라인 정규화로 핵심만 남김
- 농약/비료: '제품명 [제형/희석] [배수/농도/용량]' 형태
- 메모: '작업/품질·수치/생산량/계획' 핵심 파편을 ' · '로 요약
"""
from typing import Optional, Dict, List
import os, re

# LLM이 없어도 동작하도록 import 실패 허용
try:
    from pydantic import BaseModel, Field
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatOpenAI = None
    ChatPromptTemplate = None
    class BaseModel:  # 최소 더미
        def __init__(self, **kwargs): pass
        def dict(self): return {}
    def Field(*args, **kwargs): return None

# ──────────────────────────────────────────────────────────────────────────────
# 출력 스키마
# ──────────────────────────────────────────────────────────────────────────────
class CsvNormalized(BaseModel):
    site: Optional[str] = Field(default=None)
    crop: Optional[str] = Field(default=None)
    operation: Optional[str] = Field(default=None)  # 파종·정식 / 재배관리 / 병해충관리 / 수확 / 출하·유통
    pesticide: Optional[str] = Field(default=None)
    fertiliser: Optional[str] = Field(default=None)
    memo: Optional[str] = Field(default=None)

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트 (few-shot)
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM = """
너는 농작업 STT CSV를 '정확하고 간결한 6필드'로 정규화하는 도우미다.

[출력 스키마]
- site, crop, operation, pesticide, fertiliser, memo (문자열 또는 null)
- 출력은 스키마에 '딱 맞게'만 반환한다.

[정규화 규칙]
1) site: 핵심만(예: '포장-2', '하우스-3', 'A블록', '과원', '경북 상주시').
2) crop: 표준 작물명(샤인머스캣/사과/감귤/배추 등).
3) operation: ['파종·정식','재배관리','병해충관리','수확','출하·유통'] 중 하나로 매핑.
4) pesticide:
   - 금일 미사용 표현(안 쳤어/없음/미사용/오늘은 안 함 등)은 null
   - 사용 시 '제품명 + (배수/농도/용량) + [제형/희석]' 핵심만
5) fertiliser:
   - 금일 미사용은 null
   - 사용 시 '제품명 + 용량' 핵심만(예: '백두산 A형 5kg'); '비료' 단어는 제외
6) memo:
   - 농업 관련 핵심을 ' · '로 묶은 1줄 요약
   - 우선순위: [작업/품질·수치/생산량/계획] (있을 때만 포함, 최대 5개)
   - 비농업(여행/레시피/엔터/건강/일상) 내용이면 null
"""

_USER_EX1 = """
[Q&A 원문]
- 재배지(site): 제주도 서귀포에 샤인머스캣 농장 이에요.
- 작물(crop): 샤인머스캣입니다.
- 작업(operation): 보자.. 오늘은 포도가 잘 익어서 수확했어요.
- 농약(pesticide): 농약은 보자.. HAL-900 1000배 희석액 뿌렸어요.
- 비료(fertiliser): 백두산 A형 비료 5kg 줬어요.
- 메모(memo): 올해도 샤인머스켓이 아주 잘 익었어요. 날씨가 좋아서 그런지, 당도가 엄청 높아요. 수확량도 많고, 농약도 잘 듣고 있어요. 다음 주에는 포도즙을 만들 계획이에요.
"""

_ASSIST_EX1 = """
{ "site":"제주도 서귀포", "crop":"샤인머스캣", "operation":"수확", "pesticide":"HAL-900 1000배 희석액", "fertiliser":"백두산 A형 5kg", "memo":"샤인머스캣 · 수확 · 당도 높음 · 수확량 많음 · 포도즙 계획" }
""".strip()

_USER_EX2 = """
[Q&A 원문]
- 재배지(site): 경북 상주시요.
- 작물(crop): 사과나무지.
- 작업(operation): 가지치기 했지.
- 농약(pesticide): 응, 농약 안 쳤어.
- 비료(fertiliser): 비료는 전에 줬고 오늘은 안 줬어.
- 메모(memo): 사과 가지치기 마무리. 수세 안정 확인. 다음 주 칼슘 엽면시비 예정.
"""

_ASSIST_EX2 = """
{ "site":"경북 상주시", "crop":"사과", "operation":"재배관리", "pesticide":null, "fertiliser":null, "memo":"사과 · 재배관리 · 다음 주 칼슘 엽면시비 예정" }
""".strip()

_USER_TMPL = """
[Q&A 원문]
- 재배지(site): {site}
- 작물(crop): {crop}
- 작업(operation): {operation}
- 농약(pesticide): {pesticide}
- 비료(fertiliser): {fertiliser}
- 메모(memo): {memo}

위 자료를 규칙대로 정규화하여 6필드만 반환하라.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 규칙 기반 후처리 (LLM 유무와 관계없이 사용)
# ──────────────────────────────────────────────────────────────────────────────
FORMULATION_TOKENS = [
    "희석액","희석","수화제","유제","입상수화제","액상수화제","WDG","EC","SC","WP","SL","입상","액제"
]
UNIT_PAT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:kg|g|L|ℓ|ml|%)\b", re.IGNORECASE)
DILUTION_PAT = re.compile(r"\b\d+(?:\.\d+)?\s*배(?:\s*희석)?(?:\s*희석액)?\b")
YIELD_MORE_PAT = re.compile(r"(수확량|생산량|량)\s*(도\s*)?(많|높|풍부|증가)", re.IGNORECASE)
PLAN_PAT = re.compile(r"(계획|예정|할\s*예정|만들\s*계획|준비중|준비\s*중)", re.IGNORECASE)

FILLER_LEADING = re.compile(
    r"^\s*(농약|비료)\s*(은|는|:)?\s*(보자\.\.|보자|자|음|어|아|응|아뇨|아니|그|음\.\.|…|\.\.)*\s*",
    re.IGNORECASE)
FILLER_VERBS = re.compile(
    r"(뿌렸(어|어요)?|살포(함|했|했습니다|했어요)?|줬(어|어요)?|주었(어|어요)?|쳤(어|어요)?|사용(함|했|했습니다|했어요)?)$",
    re.IGNORECASE)

NEG_TODAY_PAT = re.compile(
    r"(없음|미사용|안\s*(줬|주|쳤|치|했|함|씀|썼))|((오늘|이번(엔|에는)?)\s*안\s*(줬|주|쳤|치|했|함|씀|썼))|(전\s*(에|엔)|지난번)\s*(줬|주었|쳤|치었)",
    re.IGNORECASE
)

NAME_STOPWORDS = {"비료"}

def _strip_leading_and_verbs(s: str) -> str:
    s = FILLER_LEADING.sub("", s or "")
    s = s.strip()
    s = re.sub(r"[\.…]+$", "", s)  # 말줄임표/마침표 제거
    s = FILLER_VERBS.sub("", s).strip()
    return re.sub(r"\s{2,}", " ", s)

def _ensure_contains_formulation(raw: str, normalized: str) -> str:
    if not raw or not normalized:
        return normalized
    raw_has = [t for t in FORMULATION_TOKENS if t in raw]
    if raw_has and not any(t in normalized for t in FORMULATION_TOKENS):
        tok = "희석액" if any("희석" in t for t in raw_has) else raw_has[0]
        if tok not in normalized:
            return (normalized + " " + tok).strip()
    return normalized

def _dedup_facets(text: str, joiner: str = " · ") -> str:
    if not text:
        return text
    parts: List[str] = [p.strip() for p in re.split(r"\s*[·|/,-]\s*", text) if p.strip()]
    out: List[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return joiner.join(out)

def _ensure_memo_facets(memo_norm: Optional[str], raw_memo: Optional[str]) -> Optional[str]:
    if memo_norm is None and not raw_memo:
        return None
    memo = memo_norm or ""
    if raw_memo and YIELD_MORE_PAT.search(raw_memo) and "수확량 많음" not in memo:
        memo = (memo + (" · " if memo else "") + "수확량 많음").strip()
    if raw_memo and PLAN_PAT.search(raw_memo):
        target = None
        m = re.search(r"(포도즙|즙|주스|와인|액비|퇴비|출하|선별|건조|저장|유통)", raw_memo)
        if m:
            noun = m.group(1)
            target = "포도즙" if noun == "즙" else noun
        facet = (target + " 계획") if target else "작업 계획"
        if facet not in (memo or ""):
            memo = (memo + (" · " if memo else "") + facet).strip()
    memo = _dedup_facets(memo)
    return memo or None

def _null_if_negative_today(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if NEG_TODAY_PAT.search(s):
        return None
    return s

def _extract_core_name(tokens: List[str]) -> str:
    core: List[str] = []
    for t in tokens:
        if (UNIT_PAT.search(t) or DILUTION_PAT.search(t) or (t in FORMULATION_TOKENS)):
            break
        if t in NAME_STOPWORDS:
            continue
        core.append(t)
    return " ".join(core).strip()

def _extract_core_chem(raw: str, kind: str) -> Optional[str]:
    """
    kind ∈ {'pesticide','fertiliser'}
    입력 raw에서 핵심요소만 구성: 제품명 [제형/희석] [배수/농도/용량]
    """
    txt = (raw or "").strip()
    if not txt:
        return None
    txt = _strip_leading_and_verbs(txt)
    toks = [t for t in re.split(r"\s+", txt) if t]

    name = _extract_core_name(toks)
    if kind == "fertiliser" and name:
        name = re.sub(r"\b비료\b", "", name).strip()

    formulation = None
    for t in toks:
        if t in FORMULATION_TOKENS:
            formulation = t
            break

    dilution = None
    m = DILUTION_PAT.search(txt)
    if m:
        dilution = m.group(0).strip()

    amount = None
    m2 = UNIT_PAT.search(txt)
    if m2:
        amount = m2.group(0).strip()

    parts = []
    if name:
        parts.append(name)
    if formulation and (formulation not in (dilution or "")):
        parts.append(formulation)
    if dilution:
        parts.append(dilution)
    if amount:
        parts.append(amount)

    out = " ".join(parts).strip()
    return out or None

def _llm_available() -> bool:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())
    return bool(key and ChatOpenAI and ChatPromptTemplate)

# ──────────────────────────────────────────────────────────────────────────────
# 실행 함수
# ──────────────────────────────────────────────────────────────────────────────
def normalize_csv_semantic(qa: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    qa: {"site":..,"crop":..,"operation":..,"pesticide":..,"fertiliser":..,"memo":..}
    return: 동일 키의 정규화된 dict (후처리 보강 포함)
    """
    # 기본값: 입력값 복사
    base = {
        "site": qa.get("site"),
        "crop": qa.get("crop"),
        "operation": qa.get("operation"),
        "pesticide": qa.get("pesticide"),
        "fertiliser": qa.get("fertiliser"),
        "memo": qa.get("memo"),
    }

    # ── 1) LLM 경로 시도 ──────────────────────────────────────────────────────
    if _llm_available():
        try:
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", _SYSTEM),
                ("user", _USER_EX1),
                ("assistant", _ASSIST_EX1),
                ("user", _USER_EX2),
                ("assistant", _ASSIST_EX2),
                ("user", _USER_TMPL),
            ])
            structured = llm.with_structured_output(CsvNormalized)
            msg = prompt.invoke({
                "site": base["site"] or "",
                "crop": base["crop"] or "",
                "operation": base["operation"] or "",
                "pesticide": base["pesticide"] or "",
                "fertiliser": base["fertiliser"] or "",
                "memo": base["memo"] or "",
            })
            out: CsvNormalized = structured.invoke(msg)
            result = {
                "site": out.site,
                "crop": out.crop,
                "operation": out.operation,
                "pesticide": out.pesticide,
                "fertiliser": out.fertiliser,
                "memo": out.memo,
            }
        except Exception:
            result = dict(base)  # LLM 실패 → 오프라인 규칙으로 계속
    else:
        result = dict(base)

    # ── 2) 후처리/보강 (LLM 유무와 무관) ──────────────────────────────────────
    # 농약/비료: 금일 미사용이면 null
    result["pesticide"]  = _null_if_negative_today(result.get("pesticide")  or base.get("pesticide")  or "")
    result["fertiliser"] = _null_if_negative_today(result.get("fertiliser") or base.get("fertiliser") or "")

    # 핵심 재구성
    if result.get("pesticide"):
        core = _extract_core_chem(base.get("pesticide") or result["pesticide"], "pesticide")
        # 제형/희석 누락 보강
        core = _ensure_contains_formulation(base.get("pesticide") or "", core or "")
        result["pesticide"] = core or None

    if result.get("fertiliser"):
        core = _extract_core_chem(base.get("fertiliser") or result["fertiliser"], "fertiliser")
        result["fertiliser"] = core or None

    # 메모 파편 보강/중복제거
    raw_memo = base.get("memo") or ""
    result["memo"] = _ensure_memo_facets(result.get("memo"), raw_memo)
    if result.get("memo"):
        result["memo"] = re.sub(r"\s*·\s*", " · ", result["memo"]).strip(" ·")

    return result
