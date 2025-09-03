# -*- coding: utf-8 -*-
"""
Enhanced Parser Evaluation System
- Stanza를 활용한 구문분석 성능 평가
- GPT-4o, Claude를 활용한 부사절 기반 문장 재작성
- LAS, UAS, ClauseSpanF1 등 다양한 메트릭 평가
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stanza import
try:
    import stanza
    STANZA_AVAILABLE = True
    logger.info("Stanza is available")
except ImportError:
    STANZA_AVAILABLE = False
    logger.warning("Stanza not available. Install with: pip install stanza")

# OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI is available")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with: pip install openai")

# Anthropic import
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
    logger.info("Anthropic is available")
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available. Install with: pip install anthropic")

from dep_utils import parse_conllu_lines, uas_las, clause_span_f1

class ParserEvaluator:
    """구문분석기 평가 클래스"""
    
    def __init__(self, stanza_lang: str = "ko"):
        self.stanza_lang = stanza_lang
        self.nlp_stanza = None
        
        # Stanza 모델 로드
        if STANZA_AVAILABLE:
            try:
                self.nlp_stanza = stanza.Pipeline(lang=stanza_lang, processors='tokenize,pos,lemma,depparse')
                logger.info(f"Loaded Stanza model for language: {stanza_lang}")
            except Exception as e:
                logger.warning(f"Failed to load Stanza model: {e}")
    
    def stanza_parse(self, text: str) -> List[Dict]:
        """Stanza를 사용한 구문분석"""
        if not self.nlp_stanza:
            return []
        
        doc = self.nlp_stanza(text)
        rows = []
        
        for sent in doc.sentences:
            for word in sent.words:
                rows.append({
                    "id": word.id,
                    "form": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "head": word.head,
                    "deprel": word.deprel
                })
        
        return rows
    
    def evaluate_parser(self, pred_rows: List[Dict], gold_rows: List[Dict]) -> Dict[str, float]:
        """구문분석 결과 평가"""
        if not pred_rows or not gold_rows:
            return {"UAS": 0.0, "LAS": 0.0, "ClauseSpanF1": 0.0}
        
        uas, las = uas_las(pred_rows, gold_rows)
        clause_f1 = clause_span_f1(pred_rows, gold_rows)
        
        return {
            "UAS": uas,
            "LAS": las,
            "ClauseSpanF1": clause_f1
        }

class LLMRewriter:
    """LLM을 활용한 문장 재작성 및 구문분석 클래스 (GPT + Claude)"""
    
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None, models: List[str] = None):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        
        # 기본 모델 리스트 설정 - GPT-5 제거
        if models is None:
            # self.models = ["gpt-4o", "gpt-5", "claude-3-haiku-20240307"]  # GPT-5 주석 처리
            self.models = ["gpt-4o", "claude-3-haiku-20240307"]
        else:
            # GPT-5 필터링
            self.models = [model for model in models if model != "gpt-5"]
        
        self.openai_clients = {}
        self.anthropic_client = None
        
        # OpenAI 클라이언트 초기화 - GPT-5 제외
        if OPENAI_AVAILABLE and openai_api_key:
            try:
                for model in self.models:
                    if model.startswith(("gpt-4")):  # gpt-5 제거
                        self.openai_clients[model] = openai.OpenAI(api_key=openai_api_key)
                logger.info(f"Initialized OpenAI clients for models: {[m for m in self.models if m.startswith(('gpt-4'))]}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI clients: {e}")
        
        # 모델명 매핑 - GPT-5 관련 매핑 주석 처리
        self.model_mapping = {
            # "gpt-5": "gpt-5",  # GPT-5 주석 처리
            # "gpt-5o": "gpt-5",  # gpt-5o 주석 처리
            # "gpt-5o-mini": "gpt-4o-mini",  # gpt-5o-mini 주석 처리
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini"
        }
        
        # Anthropic 클라이언트 초기화
        if ANTHROPIC_AVAILABLE and anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                logger.info("Initialized Anthropic client successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                logger.error(f"Anthropic API key provided: {bool(anthropic_api_key)}")
                logger.error(f"Anthropic library available: {ANTHROPIC_AVAILABLE}")
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("Anthropic library not available - install with: pip install anthropic")
            if not anthropic_api_key:
                logger.warning("Anthropic API key not provided")
        
        if not openai_api_key and not anthropic_api_key:
            logger.warning("No API keys provided - LLM rewriting will be disabled")
    
    def extract_clauses(self, gold_dep: str) -> List[Dict]:
        """gold_dep에서 부사절(advcl)만 추출"""
        try:
            gold_rows = parse_conllu_lines(gold_dep)
            clauses = []
            
            logger.info(f"Extracting advcl clauses from {len(gold_rows)} tokens")
            
            for row in gold_rows:
                if row["deprel"] == "advcl":  # advcl만 추출, acl:relcl 제외
                    # 절의 범위 찾기 (더 정확한 방법)
                    clause_tokens = [row]
                    
                    # 이 절에 종속된 모든 토큰들 찾기
                    for other_row in gold_rows:
                        if other_row["head"] == row["id"]:
                            clause_tokens.append(other_row)
                    
                    # 절의 텍스트 구성
                    clause_text = " ".join([t["form"] for t in sorted(clause_tokens, key=lambda x: x["id"])])
                    
                    clauses.append({
                        "head_id": row["id"],
                        "head_form": row["form"],
                        "deprel": row["deprel"],
                        "tokens": clause_tokens,
                        "text": clause_text
                    })
                    
                    logger.info(f"Found advcl clause: {clause_text}")
            
            logger.info(f"Total advcl clauses found: {len(clauses)}")
            return clauses
            
        except Exception as e:
            logger.error(f"Error extracting clauses: {e}")
            return []
    
    def rewrite_with_gpt(self, text: str, instruction: str, model: str, clause_info: Dict = None) -> str:
        """GPT 모델을 사용한 문장 재작성"""
        if model not in self.openai_clients:
            logger.warning(f"OpenAI client for model {model} not available")
            return ""
        
        try:
            # 모델명 매핑 적용
            actual_model = self.model_mapping.get(model, model)
            logger.info(f"Using mapped model name: {model} -> {actual_model}")
            
            # 모델별 다른 프롬프트와 파라미터 사용 - GPT-5 관련 코드 주석 처리
            # if model.startswith("gpt-5"):
            #     # GPT-5용 프롬프트 - 더 상세하고 구조화된 지시사항
            #     if clause_info:
            #         prompt = f"""
            # 당신은 한국어 문장 재작성 전문가입니다. 다음 지시사항을 정확히 따라주세요.
            # 
            # **원문**: {text}
            # **대상 부사절**: {clause_info["text"]}
            # **재작성 지시사항**: {instruction}
            # 
            # **요구사항**:
            # 1. 지정된 부사절 부분을 중심으로 재작성하세요
            # 2. 원문의 의미를 유지하면서 자연스럽게 표현하세요
            # 3. 문법적으로 올바른 한국어로 작성하세요
            # 4. 지시사항의 의도를 정확히 반영하세요
            # 
            # **재작성된 문장**:
            # """
            #     else:
            #         prompt = f"""
            # 당신은 한국어 문장 재작성 전문가입니다. 다음 지시사항을 정확히 따라주세요.
            # 
            # **원문**: {text}
            # **재작성 지시사항**: {instruction}
            # 
            # **요구사항**:
            # 1. 원문의 핵심 의미를 유지하세요
            # 2. 지시사항에 따라 문장 구조를 개선하세요
            # 3. 자연스럽고 읽기 쉬운 한국어로 작성하세요
            # 4. 문법적 정확성을 보장하세요
            # 
            # **재작성된 문장**:
            # """
            #     
            #     # GPT-5 모델용 파라미터 (temperature 제거, 더 긴 토큰)
            #     api_params = {
            #         "model": actual_model,
            #         "messages": [
            #             {"role": "system", "content": "당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항을 정확히 이해하고 자연스럽고 문법적으로 올바른 한국어로 재작성해주세요."},
            #             {"role": "user", "content": prompt}
            #         ],
            #         "max_completion_tokens": 800
            #     }
            # else:
            # GPT-4o용 프롬프트 - 더 간결하고 직관적인 지시사항
            if clause_info:
                prompt = f"""
다음 문장에서 부사절 부분을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
부사절: {clause_info["text"]}
지시사항: {instruction}

재작성된 문장:
"""
            else:
                prompt = f"""
다음 문장을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
지시사항: {instruction}

재작성된 문장:
"""
            
            # GPT-4o 모델용 파라미터 (temperature 포함, 더 짧은 토큰)
            api_params = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": "당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항에 따라 문장을 자연스럽게 재작성해주세요."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            # OpenAI 1.0+ 버전용 API 호출
            response = self.openai_clients[model].chat.completions.create(**api_params)
            
            result = response.choices[0].message.content.strip()
            logger.info(f"GPT {model} rewrite successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"GPT {model} API 호출 중 오류: {e}")
            return ""
    
    def rewrite_with_claude(self, text: str, instruction: str, model: str, clause_info: Dict = None) -> str:
        """Claude 모델을 사용한 문장 재작성"""
        if not self.anthropic_client:
            logger.warning("Anthropic client not available")
            return ""
        
        try:
            if clause_info:
                clause_type = "부사절" if clause_info["deprel"] == "advcl" else "관형절"
                prompt = f"""
당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항에 따라 문장을 자연스럽게 재작성해주세요.

다음 문장에서 {clause_type} 부분을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
{clause_type}: {clause_info["text"]}
지시사항: {instruction}

재작성된 문장:
"""
            else:
                prompt = f"""
당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항에 따라 문장을 자연스럽게 재작성해주세요.

다음 문장을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
지시사항: {instruction}

재작성된 문장:
"""
            
            # Anthropic API 호출
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            logger.info(f"Claude {model} rewrite successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Claude {model} API 호출 중 오류: {e}")
            return ""
    
    def rewrite_with_llm(self, text: str, instruction: str, model: str, clause_info: Dict = None) -> str:
        """모델 타입에 따라 적절한 API 호출"""
        if model.startswith(("gpt-4")):  # gpt-5 제거
            return self.rewrite_with_gpt(text, instruction, model, clause_info)
        elif model.startswith("claude"):
            return self.rewrite_with_claude(text, instruction, model, clause_info)
        else:
            logger.warning(f"Unknown model type: {model}")
            return ""
    
    def rewrite_with_all_models(self, text: str, instruction: str, clause_info: Dict = None) -> Dict[str, str]:
        """모든 사용 가능한 모델로 문장 재작성"""
        results = {}
        
        for model in self.models:
            if model.startswith(("gpt-4")) and model in self.openai_clients:  # gpt-5 제거
                result = self.rewrite_with_gpt(text, instruction, model, clause_info)
                results[model] = result
            elif model.startswith("claude") and self.anthropic_client:
                result = self.rewrite_with_claude(text, instruction, model, clause_info)
                results[model] = result
            else:
                logger.warning(f"Model {model} not available")
                results[model] = ""
        
        return results

    def parse_dependency_with_gpt(self, text: str, model: str) -> List[Dict]:
        """GPT 모델을 사용한 보편의존구문분석"""
        if model not in self.openai_clients:
            logger.warning(f"OpenAI client for model {model} not available")
            return []
        
        try:
            # 모델명 매핑 적용
            actual_model = self.model_mapping.get(model, model)
            logger.info(f"Using mapped model name (dependency): {model} -> {actual_model}")
            
            # 모델별 다른 프롬프트 사용 - GPT-5 관련 코드 주석 처리
            # if model.startswith("gpt-5"):
            #     # GPT-5용 프롬프트 - 더 상세하고 구조화된 지시사항
            #     prompt = f"""
            # 당신은 한국어 보편의존구문분석 전문가입니다. 다음 지시사항을 정확히 따라주세요.
            # 
            # **분석 대상 문장**: {text}
            # 
            # **분석 요구사항**:
            # 1. 각 단어를 개별 토큰으로 분리하여 분석하세요
            # 2. 한국어의 특성을 고려하여 정확한 의존관계를 파악하세요
            # 3. 문장의 핵심 동사/형용사를 루트로 설정하세요
            # 4. 보편의존구문분석 표준을 준수하세요
            # 
            # **출력 형식** (CoNLL-U):
            # ID	FORM	LEMMA	UPOS	XPOS	FEATS	HEAD	DEPREL	DEPS	MISC
            # 
            # **주의사항**:
            # - ID: 토큰의 순서 (1부터 시작)
            # - FORM: 실제 단어 형태
            # - LEMMA: 기본형 (동사/형용사의 경우)
            # - UPOS: 보편품사태그 (NOUN, VERB, ADJ, ADV, ADP, DET, PRON, NUM, PUNCT 등)
            # - HEAD: 의존관계의 머리 토큰 ID (루트는 0)
            # - DEPREL: 의존관계 라벨 (nsubj, obj, advmod, root 등)
            # 
            # **분석 결과**:
            # """
            # else:
            # GPT-4o용 프롬프트 - 더 간결한 지시사항
            prompt = f"""
다음 한국어 문장의 보편의존구문분석(Universal Dependency)을 수행해주세요.

문장: {text}

각 단어에 대해 다음 형식으로 분석해주세요:
1. ID (토큰 순서)
2. FORM (단어 형태)
3. LEMMA (기본형)
4. UPOS (보편품사태그)
5. HEAD (의존관계의 머리 토큰 ID)
6. DEPREL (의존관계 라벨)

CoNLL-U 형식으로 출력해주세요:
1	단어1	기본형1	품사1	_	_	2	관계1	_	_
2	단어2	기본형2	품사2	_	_	0	root	_	_

주의사항:
- HEAD가 0인 토큰은 루트(문장의 핵심 동사/형용사)
- 한국어의 특성을 고려하여 정확한 의존관계를 분석
- 각 토큰은 공백으로 구분된 단어 단위로 분석
"""
            
            # 모델별 파라미터 설정 - GPT-5 관련 코드 주석 처리
            # if model.startswith("gpt-5"):
            #     api_params = {
            #         "model": actual_model,
            #         "messages": [
            #             {"role": "system", "content": "당신은 한국어 보편의존구문분석 전문가입니다. 주어진 문장의 의존구문분석을 정확하고 상세하게 수행해주세요."},
            #             {"role": "user", "content": prompt}
            #         ],
            #         "max_completion_tokens": 1200
            #     }
            # else:
            api_params = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": "당신은 한국어 보편의존구문분석 전문가입니다. 주어진 문장의 의존구문분석을 정확하게 수행해주세요."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = self.openai_clients[model].chat.completions.create(**api_params)
            result = response.choices[0].message.content.strip()
            
            # CoNLL-U 형식 파싱
            parsed_rows = self.parse_conllu_response(result)
            logger.info(f"GPT {model} dependency parsing successful: {len(parsed_rows)} tokens")
            return parsed_rows
            
        except Exception as e:
            logger.error(f"GPT {model} dependency parsing error: {e}")
            return []
    
    def parse_dependency_with_claude(self, text: str, model: str) -> List[Dict]:
        """Claude 모델을 사용한 보편의존구문분석"""
        if not self.anthropic_client:
            logger.warning("Anthropic client not available")
            return []
        
        try:
            prompt = f"""
당신은 한국어 보편의존구문분석 전문가입니다. 주어진 문장의 의존구문분석을 정확하게 수행해주세요.

다음 한국어 문장의 보편의존구문분석(Universal Dependency)을 수행해주세요.

문장: {text}

각 단어에 대해 다음 형식으로 분석해주세요:
1. ID (토큰 순서)
2. FORM (단어 형태)
3. LEMMA (기본형)
4. UPOS (보편품사태그)
5. HEAD (의존관계의 머리 토큰 ID)
6. DEPREL (의존관계 라벨)

CoNLL-U 형식으로 출력해주세요:
1	단어1	기본형1	품사1	_	_	2	관계1	_	_
2	단어2	기본형2	품사2	_	_	0	root	_	_

주의사항:
- HEAD가 0인 토큰은 루트(문장의 핵심 동사/형용사)
- 한국어의 특성을 고려하여 정확한 의존관계를 분석
- 각 토큰은 공백으로 구분된 단어 단위로 분석
"""
            
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            
            # CoNLL-U 형식 파싱
            parsed_rows = self.parse_conllu_response(result)
            logger.info(f"Claude {model} dependency parsing successful: {len(parsed_rows)} tokens")
            return parsed_rows
            
        except Exception as e:
            logger.error(f"Claude {model} dependency parsing error: {e}")
            return []
    
    def parse_dependency_with_llm(self, text: str, model: str) -> List[Dict]:
        """모델 타입에 따라 적절한 API 호출"""
        if model.startswith(("gpt-4")):  # gpt-5 제거
            return self.parse_dependency_with_gpt(text, model)
        elif model.startswith("claude"):
            return self.parse_dependency_with_claude(text, model)
        else:
            logger.warning(f"Unknown model type: {model}")
            return []
    
    def parse_conllu_response(self, response_text: str) -> List[Dict]:
        """LLM 응답을 CoNLL-U 형식으로 파싱"""
        try:
            rows = []
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # CoNLL-U 형식 파싱: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                parts = line.split('\t')
                if len(parts) >= 8:
                    try:
                        row = {
                            "id": int(parts[0]),
                            "form": parts[1],
                            "lemma": parts[2],
                            "upos": parts[3],
                            "head": int(parts[6]),
                            "deprel": parts[7]
                        }
                        rows.append(row)
                    except (ValueError, IndexError):
                        continue
            
            return rows
            
        except Exception as e:
            logger.error(f"Error parsing CoNLL-U response: {e}")
            return []

    def rewrite_with_llm_without_advcl(self, text: str, instruction: str, model: str) -> str:
        """advcl 정보 없이 문장 재작성"""
        logger.info(f"Starting rewrite without advcl for model: {model}")
        
        if model.startswith(("gpt-4")):  # gpt-5 제거
            logger.info(f"Using GPT model: {model}")
            return self.rewrite_with_gpt_without_advcl(text, instruction, model)
        elif model.startswith("claude"):
            logger.info(f"Using Claude model: {model}")
            result = self.rewrite_with_claude_without_advcl(text, instruction, model)
            logger.info(f"Claude without advcl result: {result[:50] if result else 'None'}...")
            return result
        else:
            logger.warning(f"Unknown model type: {model}")
            return ""
    
    def rewrite_with_gpt_without_advcl(self, text: str, instruction: str, model: str) -> str:
        """GPT 모델을 사용한 문장 재작성 (advcl 정보 없이)"""
        if model not in self.openai_clients:
            logger.warning(f"OpenAI client for model {model} not available")
            return ""
        
        try:
            # 모델명 매핑 적용
            actual_model = self.model_mapping.get(model, model)
            logger.info(f"Using mapped model name (without advcl): {model} -> {actual_model}")
            
            # advcl 정보 없이 재작성하는 프롬프트 - GPT-5 관련 코드 주석 처리
            # if model.startswith("gpt-5"):
            #     prompt = f"""
            # 당신은 한국어 문장 재작성 전문가입니다. 다음 지시사항을 정확히 따라주세요.
            # 
            # **원문**: {text}
            # **재작성 지시사항**: {instruction}
            # 
            # **요구사항**:
            # 1. 원문의 핵심 의미를 유지하세요
            # 2. 지시사항에 따라 문장 구조를 개선하세요
            # 3. 자연스럽고 읽기 쉬운 한국어로 작성하세요
            # 4. 문법적 정확성을 보장하세요
            # 
            # **재작성된 문장**:
            # """
            #     
            #     # GPT-5 모델용 파라미터
            #     api_params = {
            #         "model": actual_model,
            #         "messages": [
            #             {"role": "system", "content": "당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항을 정확히 이해하고 자연스럽고 문법적으로 올바른 한국어로 재작성해주세요."},
            #             {"role": "user", "content": prompt}
            #         ],
            #         "max_completion_tokens": 800
            #     }
            # else:
            prompt = f"""
다음 문장을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
지시사항: {instruction}

재작성된 문장:
"""
            
            # GPT-4o 모델용 파라미터
            api_params = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": "당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항에 따라 문장을 자연스럽게 재작성해주세요."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            # OpenAI 1.0+ 버전용 API 호출
            response = self.openai_clients[model].chat.completions.create(**api_params)
            
            result = response.choices[0].message.content.strip()
            logger.info(f"GPT {model} rewrite without advcl successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"GPT {model} API 호출 중 오류: {e}")
            return ""
    
    def rewrite_with_claude_without_advcl(self, text: str, instruction: str, model: str) -> str:
        """Claude 모델을 사용한 문장 재작성 (advcl 정보 없이)"""
        if not self.anthropic_client:
            logger.warning("Anthropic client not available")
            return ""
        
        try:
            logger.info(f"Starting Claude without advcl rewrite for model: {model}")
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Instruction: {instruction}")
            
            prompt = f"""
당신은 한국어 문장 재작성 전문가입니다. 주어진 지시사항에 따라 문장을 자연스럽게 재작성해주세요.

다음 문장을 주어진 지시사항에 따라 재작성하세요.

원문: {text}
지시사항: {instruction}

재작성된 문장:
"""
            
            logger.info(f"Sending request to Claude API for model: {model}")
            
            # Anthropic API 호출
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text.strip()
            logger.info(f"Claude {model} rewrite without advcl successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Claude {model} API 호출 중 오류: {e}")
            logger.error(f"Error details: {str(e)}")
            return ""

class EnhancedEvaluationSystem:
    """향상된 평가 시스템"""
    
    def __init__(self, openai_api_key: str = None, anthropic_api_key: str = None, llm_models: List[str] = None):
        self.parser_evaluator = ParserEvaluator()
        self.llm_rewriter = LLMRewriter(openai_api_key, anthropic_api_key, llm_models) if (openai_api_key or anthropic_api_key) else None
        
        if not openai_api_key and not anthropic_api_key:
            logger.warning("No API keys provided - LLM rewriting will be disabled")
        
    def evaluate_single_example(self, example: Dict) -> Dict[str, Any]:
        """단일 예제 평가"""
        result = {
            "id": example.get("id", ""),
            "text": example.get("text", ""),
            "stanza_results": {},
            "llm_dependency_results": {},
            "llm_rewrite_results": {}
        }
        
        text = example.get("text", "")
        gold_dep = example.get("gold_dep", "")
        rewrite_instruction = example.get("rewrite_instruction", "")
        gold_rewrite = example.get("gold_rewrite", "")
        
        logger.info(f"Evaluating example {result['id']}: {text[:50]}...")
        
        if not text or not gold_dep:
            logger.warning(f"Missing text or gold_dep for example {result['id']}")
            return result
        
        # Gold standard 파싱
        try:
            gold_rows = parse_conllu_lines(gold_dep)
            logger.info(f"Parsed {len(gold_rows)} gold tokens")
        except Exception as e:
            logger.error(f"Error parsing gold_dep: {e}")
            return result
        
        # Stanza 평가
        if STANZA_AVAILABLE and self.parser_evaluator.nlp_stanza:
            stanza_rows = self.parser_evaluator.stanza_parse(text)
            if stanza_rows:
                stanza_metrics = self.parser_evaluator.evaluate_parser(stanza_rows, gold_rows)
                result["stanza_results"] = {
                    "metrics": stanza_metrics,
                    "parsed_rows": stanza_rows
                }
                logger.info(f"Stanza metrics: {stanza_metrics}")
            else:
                logger.warning("Stanza parsing returned empty results")
        
        # LLM 의존구문분석 평가
        if self.llm_rewriter:
            logger.info("Starting LLM dependency parsing evaluation")
            
            llm_dependency_results = {}
            
            for model in self.llm_rewriter.models:
                logger.info(f"Parsing dependency with {model}")
                
                # LLM으로 의존구문분석 수행
                llm_parsed_rows = self.llm_rewriter.parse_dependency_with_llm(text, model)
                
                if llm_parsed_rows:
                    # gold_dep과 비교하여 UAS, LAS 계산
                    llm_metrics = self.parser_evaluator.evaluate_parser(llm_parsed_rows, gold_rows)
                    
                    llm_dependency_results[model] = {
                        "metrics": llm_metrics,
                        "parsed_rows": llm_parsed_rows
                    }
                    
                    logger.info(f"LLM {model} dependency metrics: {llm_metrics}")
                else:
                    logger.warning(f"LLM {model} dependency parsing returned empty results")
                    llm_dependency_results[model] = {
                        "metrics": {"UAS": 0.0, "LAS": 0.0, "ClauseSpanF1": 0.0},
                        "parsed_rows": []
                    }
            
            result["llm_dependency_results"] = llm_dependency_results
        
        # LLM 재작성 평가 (기존 코드)
        if self.llm_rewriter and rewrite_instruction and gold_rewrite:
            logger.info("Starting LLM rewrite evaluation with all models")
            
            # 절 추출 (부사절 + 관형절)
            clauses = self.llm_rewriter.extract_clauses(gold_dep)
            
            # 각 모델별 결과 저장
            model_results = {}
            
            for model in self.llm_rewriter.models:
                model_results[model] = {
                    "clauses": clauses,
                    "rewrite_results": [],
                    "rewrite_results_without_advcl": [],  # advcl 정보 없이 재작성한 결과
                    "best_f1_score": 0.0,
                    "best_rewritten_text": "",
                    "avg_f1_score": 0.0,
                    "best_f1_score_without_advcl": 0.0,  # advcl 정보 없이 재작성한 최고 F1
                    "best_rewritten_text_without_advcl": "",  # advcl 정보 없이 재작성한 최고 텍스트
                    "avg_f1_score_without_advcl": 0.0  # advcl 정보 없이 재작성한 평균 F1
                }
                
                rewrite_results = []
                rewrite_results_without_advcl = []  # advcl 정보 없이 재작성한 결과
                
                # 각 절에 대해 재작성 시도 (부사절이 있는 경우)
                for clause in clauses:
                    logger.info(f"Rewriting clause with {model}: {clause['deprel']} - {clause['text']}")
                    
                    rewritten = self.llm_rewriter.rewrite_with_llm(
                        text, 
                        rewrite_instruction, 
                        model,
                        clause
                    )
                    
                    if rewritten:
                        # F1 점수 계산
                        f1_score = self.calculate_text_similarity(rewritten, gold_rewrite)
                        
                        rewrite_results.append({
                            "clause_type": clause["deprel"],
                            "clause_text": clause["text"],
                            "rewritten_text": rewritten,
                            "f1_score": f1_score,
                            "gold_rewrite": gold_rewrite
                        })
                        
                        logger.info(f"Clause rewrite F1 ({model}): {f1_score}")
                
                # 전체 문장 재작성도 항상 시도 (부사절이 있든 없든)
                logger.info(f"Attempting full sentence rewrite with {model}")
                full_rewritten = self.llm_rewriter.rewrite_with_llm(text, rewrite_instruction, model)
                if full_rewritten:
                    full_f1_score = self.calculate_text_similarity(full_rewritten, gold_rewrite)
                    rewrite_results.append({
                        "clause_type": "full_sentence",
                        "clause_text": text,
                        "rewritten_text": full_rewritten,
                        "f1_score": full_f1_score,
                        "gold_rewrite": gold_rewrite
                    })
                    logger.info(f"Full sentence rewrite F1 ({model}): {full_f1_score}")
                
                # advcl 정보 없이 재작성 시도 (새로운 방식)
                logger.info(f"Attempting rewrite without advcl information with {model}")
                rewritten_without_advcl = self.llm_rewriter.rewrite_with_llm_without_advcl(
                    text, 
                    rewrite_instruction, 
                    model
                )
                
                logger.info(f"Rewrite without advcl result for {model}: {rewritten_without_advcl[:50] if rewritten_without_advcl else 'None'}...")
                
                if rewritten_without_advcl:
                    f1_score_without_advcl = self.calculate_text_similarity(rewritten_without_advcl, gold_rewrite)
                    rewrite_results_without_advcl.append({
                        "clause_type": "full_sentence_without_advcl",
                        "clause_text": text,
                        "rewritten_text": rewritten_without_advcl,
                        "f1_score": f1_score_without_advcl,
                        "gold_rewrite": gold_rewrite
                    })
                    logger.info(f"Rewrite without advcl F1 ({model}): {f1_score_without_advcl}")
                else:
                    logger.warning(f"No rewrite result without advcl for model {model}")
                
                # 최고 F1 점수 선택 (advcl 정보 포함)
                best_result = max(rewrite_results, key=lambda x: x["f1_score"]) if rewrite_results else None
                
                # 최고 F1 점수 선택 (advcl 정보 없음)
                best_result_without_advcl = max(rewrite_results_without_advcl, key=lambda x: x["f1_score"]) if rewrite_results_without_advcl else None
                
                logger.info(f"Best result without advcl for {model}: {best_result_without_advcl['f1_score'] if best_result_without_advcl else 'None'}")
                
                model_results[model].update({
                    "rewrite_results": rewrite_results,
                    "rewrite_results_without_advcl": rewrite_results_without_advcl,
                    "best_f1_score": best_result["f1_score"] if best_result else 0.0,
                    "best_rewritten_text": best_result["rewritten_text"] if best_result else "",
                    "avg_f1_score": np.mean([r["f1_score"] for r in rewrite_results]) if rewrite_results else 0.0,
                    "best_f1_score_without_advcl": best_result_without_advcl["f1_score"] if best_result_without_advcl else 0.0,
                    "best_rewritten_text_without_advcl": best_result_without_advcl["rewritten_text"] if best_result_without_advcl else "",
                    "avg_f1_score_without_advcl": np.mean([r["f1_score"] for r in rewrite_results_without_advcl]) if rewrite_results_without_advcl else 0.0
                })
                
                logger.info(f"LLM {model} rewrite completed. Best F1 (with advcl): {model_results[model]['best_f1_score']}, Best F1 (without advcl): {model_results[model]['best_f1_score_without_advcl']}")
                logger.info(f"LLM {model} rewrite results count (with advcl): {len(rewrite_results)}, (without advcl): {len(rewrite_results_without_advcl)}")
            
            result["llm_rewrite_results"] = model_results
            
        else:
            logger.warning("LLM rewriting skipped - missing API key, instruction, or gold_rewrite")
        
        return result
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """텍스트 유사도 계산 (토큰 기반 F1)"""
        try:
            # 한국어 토큰화 (공백 기준)
            tokens1 = set(text1.strip().split())
            tokens2 = set(text2.strip().split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = len(tokens1 & tokens2)
            precision = intersection / len(tokens1)
            recall = intersection / len(tokens2)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0

    def evaluate_dataset(self, jsonl_path: str, output_path: str):
        """전체 데이터셋 평가"""
        results = []
        
        # 데이터 로드
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            logger.info(f"Loaded {len(data)} examples")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return
        
        # 각 예제 평가
        for example in tqdm(data, desc="Evaluating examples"):
            result = self.evaluate_single_example(example)
            results.append(result)
        
        # 결과 저장
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 상세 결과 저장
        try:
            with open(output_dir / "detailed_results.jsonl", 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logger.info(f"Detailed results saved to {output_dir / 'detailed_results.jsonl'}")
        except Exception as e:
            logger.error(f"Error saving detailed results: {e}")
        
        # 요약 통계 생성
        self.generate_summary_statistics(results, output_dir)
        
        logger.info(f"Evaluation completed. Results saved to {output_path}")
    
    def generate_summary_statistics(self, results: List[Dict], output_dir: Path):
        """요약 통계 생성"""
        summary = {
            "total_examples": len(results),
            "stanza_stats": {},
            "llm_dependency_stats": {},
            "llm_rewrite_stats": {}
        }
        
        # Stanza 통계
        stanza_metrics = []
        for result in results:
            if result["stanza_results"] and result["stanza_results"].get("metrics"):
                stanza_metrics.append(result["stanza_results"]["metrics"])
        
        if stanza_metrics:
            df_stanza = pd.DataFrame(stanza_metrics)
            summary["stanza_stats"] = {
                "count": len(stanza_metrics),
                "mean": df_stanza.mean().to_dict(),
                "std": df_stanza.std().to_dict()
            }
            logger.info(f"Stanza statistics: {len(stanza_metrics)} samples")
        
        # LLM 의존구문분석 통계
        llm_dependency_stats = {}
        for result in results:
            if result.get("llm_dependency_results"):
                for model, model_result in result["llm_dependency_results"].items():
                    if model_result.get("metrics"):
                        if model not in llm_dependency_stats:
                            llm_dependency_stats[model] = []
                        llm_dependency_stats[model].append(model_result["metrics"])
        
        for model, metrics_list in llm_dependency_stats.items():
            if metrics_list:
                df_llm = pd.DataFrame(metrics_list)
                summary["llm_dependency_stats"][model] = {
                    "count": len(metrics_list),
                    "mean": df_llm.mean().to_dict(),
                    "std": df_llm.std().to_dict()
                }
                logger.info(f"LLM {model} dependency statistics: {len(metrics_list)} samples")
        
        # LLM 재작성 통계 - 수정된 부분
        llm_rewrite_stats = {}
        for result in results:
            if result.get("llm_rewrite_results"):
                for model, model_result in result["llm_rewrite_results"].items():
                    # best_f1_score가 0보다 큰 경우만 포함 (실제 재작성이 수행된 경우)
                    if model_result.get("best_f1_score", 0.0) > 0.0:
                        if model not in llm_rewrite_stats:
                            llm_rewrite_stats[model] = []
                        llm_rewrite_stats[model].append(model_result["best_f1_score"])
                    # 또는 avg_f1_score가 있는 경우
                    elif model_result.get("avg_f1_score", 0.0) > 0.0:
                        if model not in llm_rewrite_stats:
                            llm_rewrite_stats[model] = []
                        llm_rewrite_stats[model].append(model_result["avg_f1_score"])
                    # rewrite_results가 있는 경우
                    elif model_result.get("rewrite_results") and len(model_result["rewrite_results"]) > 0:
                        f1_scores = [r.get("f1_score", 0.0) for r in model_result["rewrite_results"] if r.get("f1_score", 0.0) > 0.0]
                        if f1_scores:
                            if model not in llm_rewrite_stats:
                                llm_rewrite_stats[model] = []
                            llm_rewrite_stats[model].extend(f1_scores)
        
        # 각 모델별 통계 계산
        for model, f1_scores in llm_rewrite_stats.items():
            if f1_scores:
                summary["llm_rewrite_stats"][model] = {
                    "count": len(f1_scores),
                    "mean_f1_score": float(np.mean(f1_scores)),
                    "std_f1_score": float(np.std(f1_scores)),
                    "min_f1_score": float(np.min(f1_scores)),
                    "max_f1_score": float(np.max(f1_scores))
                }
                logger.info(f"LLM {model} rewrite statistics: {len(f1_scores)} samples, mean F1: {np.mean(f1_scores):.4f}")
            else:
                # 결과가 없는 경우에도 빈 통계 추가
                summary["llm_rewrite_stats"][model] = {
                    "count": 0,
                    "mean_f1_score": 0.0,
                    "std_f1_score": 0.0,
                    "min_f1_score": 0.0,
                    "max_f1_score": 0.0
                }
                logger.warning(f"LLM {model} rewrite statistics: no valid results")
        
        # 모든 모델이 llm_rewrite_stats에 포함되도록 보장
        if self.llm_rewriter:
            for model in self.llm_rewriter.models:
                if model not in summary["llm_rewrite_stats"]:
                    summary["llm_rewrite_stats"][model] = {
                        "count": 0,
                        "mean_f1_score": 0.0,
                        "std_f1_score": 0.0,
                        "min_f1_score": 0.0,
                        "max_f1_score": 0.0
                    }
                    logger.warning(f"LLM {model} not found in rewrite results")
        
        # 요약 저장
        try:
            with open(output_dir / "summary_statistics.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Summary statistics saved to {output_dir / 'summary_statistics.json'}")
        except Exception as e:
            logger.error(f"Error saving summary statistics: {e}")
        
        # CSV 형태로도 저장
        self.save_metrics_to_csv(results, output_dir)
    
    def save_metrics_to_csv(self, results: List[Dict], output_dir: Path):
        """메트릭을 CSV 형태로 저장"""
        csv_data = []
        
        for result in results:
            row = {"id": result["id"], "text": result["text"]}
            
            # Stanza 메트릭
            if result["stanza_results"] and result["stanza_results"].get("metrics"):
                metrics = result["stanza_results"]["metrics"]
                row.update({f"stanza_{k}": v for k, v in metrics.items()})
            
            # LLM 의존구문분석 메트릭
            if result.get("llm_dependency_results"):
                for model, model_result in result["llm_dependency_results"].items():
                    if model_result.get("metrics"):
                        metrics = model_result["metrics"]
                        row.update({f"{model}_dependency_{k}": v for k, v in metrics.items()})
            
            # LLM 재작성 메트릭
            if result.get("llm_rewrite_results"):
                for model, model_result in result["llm_rewrite_results"].items():
                    if model_result.get("best_f1_score"):
                        row[f"{model}_rewrite_best_f1_score"] = model_result["best_f1_score"]
                        row[f"{model}_rewrite_avg_f1_score"] = model_result.get("avg_f1_score", 0.0)
            
            csv_data.append(row)
        
        try:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding='utf-8')
            logger.info(f"Metrics CSV saved to {output_dir / 'metrics_summary.csv'}")
        except Exception as e:
            logger.error(f"Error saving metrics CSV: {e}")

def main():
    """메인 함수"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Enhanced Parser Evaluation System")
    parser.add_argument("--input", default="ko_gsd_train_final.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", default="dp/output/enhanced_results", help="Output directory path")
    parser.add_argument("--openai_key", help="OpenAI API key (optional)")
    parser.add_argument("--anthropic_key", help="Anthropic API key (optional)")
    parser.add_argument("--llm_models", nargs='+', default=["gpt-4o", "claude-3-haiku-20240307"], help="LLM models to use")
    
    args = parser.parse_args()
    
    # API 키 디버깅 정보 출력
    print(f"Anthropic API key provided: {bool(args.anthropic_key)}")
    if args.anthropic_key:
        print(f"Anthropic API key length: {len(args.anthropic_key)}")
        print(f"Anthropic API key starts with: {args.anthropic_key[:10]}...")
    
    # Anthropic 라이브러리 확인
    try:
        import anthropic
        print("Anthropic library is available")
    except ImportError:
        print("ERROR: Anthropic library not installed. Run: pip install anthropic")
        return
    
    # 평가 시스템 초기화
    evaluator = EnhancedEvaluationSystem(
        openai_api_key=args.openai_key, 
        anthropic_api_key=args.anthropic_key, 
        llm_models=args.llm_models
    )
    
    # 데이터셋 평가 실행
    evaluator.evaluate_dataset(args.input, args.output)

if __name__ == "__main__":
    main()