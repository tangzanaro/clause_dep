
# 부사절 제시를 통한 LLMs의 문장 변환 실험

한 줄 설명: 
Universal dependency의 ko-gsd-ud-train를 활용한 연구입니다.
문장 내 부사절을 탐지하여 명사절 또는 관형절이 내포된 같은 의미의 문장을 비교하는 연구입니다.

## ✨ Features
- 기능 1: 언어모델과 stanza의 비교를 통해 언어모델의 구문분석 정확도 평가
- 기능 2: golden_rewrite와 모델이 생성한 문장 비교
- 기능 3: 부사절(advcl) 단서로 제시한 instrucution과 제시하지 않은 instruction의 생성 성능 차이 비교

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
    --output dep_clause/results/ \
    --openai_key “OpenAI API key”\
    --anthropic_key “Claude API key”

#Run visualization
python enhanced_visualization.py
