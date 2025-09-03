
# -*- coding: utf-8 -*-

STRUCTURE_SYSTEM_PROMPT = """\
당신은 한국어 구문 분석기입니다.
입력 문장을 Universal Dependencies 형식으로 분석하여
다음 열 순서로 표를 출력하세요:
ID\tFORM\tLEMMA\tUPOS\tHEAD\tDEPREL

규칙:
- ID는 1부터 시작하는 토큰 번호
- FORM: 원형 그대로
- LEMMA: 기본형(사전형)
- UPOS: Universal POS 태그
- HEAD: 문장의 root는 0, 그 외에는 의존하는 head 토큰 ID
- DEPREL: Universal Dependencies 관계 라벨
- 지정된 출력 형식 외에 설명, 주석, 추가 텍스트를 넣지 마세요.
"""

def structure_user_prompt(sentence: str) -> str:
    return f"다음 문장을 분석하세요.\n\n문장: {sentence}"


REWRITE_SYSTEM_PROMPT = """\
당신은 문장을 구조적으로 재작성하는 전문가입니다.

조건:
- 주어진 문장을 지정된 구조 지시에 맞게 재작성하세요.
- 의미는 보존해야 하며, 시제·존대·어미 일관성을 유지해야 합니다.
- 불필요한 단어를 추가하지 말고, 맞춤법과 문법을 지키세요.
- 출력은 수정된 문장 하나만 제시하세요. 추가 설명은 쓰지 마세요.
"""

def rewrite_user_prompt(sentence: str, instruction: str) -> str:
    return f"문장: {sentence}\n지시: {instruction}\n조건: 의미 보존·어미 일관."
