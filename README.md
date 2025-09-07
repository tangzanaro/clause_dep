
# LLMs의 한국어 부사절 인식과 변환 능력 연구
 
Universal dependency의 ko-gsd-ud-train를 활용한 연구입니다.

문장 내 부사절을 탐지하여 명사절 또는 관형절이 내포된 동일한 의미의 문장으로 변환한 결과를 평가합니다.

## ✨ Features
- 절차 1: 언어모델과 stanza의 비교를 통해 언어모델의 구문분석 정확도 평가
- 절차 2: golden_rewrite와 모델이 생성한 문장 비교
- 절차 3: 부사절(advcl) 단서 제시 instruction과 비제시 instruction의 생성 성능 차이 비교

## 실험 데이터
| ID    | Description              |                           |
| ---------- | -------------------------|------------------------- |
| text | 직접 만들어 본 커스텀 앱바입니다.                |
| ---------- | -------------------------|------------------------- |
| gold_dep | 직접 만들어 본 커스텀 앱바입니다.                |
| ---------- | -------------------------|------------------------- |\


## 📦 Installation
```bash
git clone https://github.com/tangzanaro/clause_dep
cd clause_dep
pip install -r requirements.txt
```

## 📦 Run
```bash
#Run analysis
python enhanced_parser_evaluation.py \
    —-ko_gsd_train_final.jsonl \
    --output clause_dep/results/ \
    --openai_key “OpenAI API key”\
    --anthropic_key “Claude API key”

#Run visualization
python enhanced_visualization.py
