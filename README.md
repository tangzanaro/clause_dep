
# LLMs의 한국어 부사절 인식과 변환 능력 연구
 
Universal dependency의 ko-gsd-ud-train를 활용한 연구입니다.

문장 내 부사절을 탐지하여 명사절 또는 관형절이 내포된 동일한 의미의 문장으로 변환한 결과를 평가합니다.

## ✨ Features
- 절차 1: 언어모델과 stanza의 비교를 통해 언어모델의 구문분석 정확도 평가
- 절차 2: golden_rewrite와 모델이 생성한 문장 비교
- 절차 3: 부사절(advcl) 단서 제시 instruction과 비제시 instruction의 생성 성능 차이 비교

## 실험 데이터
| ID    | 대상(train-s821)           |        설명             |
| ---------- | -------------------------|------------------------- |
| text | 어린이들을 <ins>위한</ins> 부대시설도 많아서 가족끼리 여행 오신 분들도 많더라구요|ko-gsd-ud-train 원문|
| gold_dep | - 생략 - |UD 구문 분석 |
| instruction | gold_dep에 부사절(advcl)이 있을 때 부사절 이외의 다른 절을 사용하거나 어순 변경, 다른 형태의 부사절로 문장을 생성한다. 문장은 최대한 자연스러워야 하지만 변환이 불가능하다 판단되는 text는 gold_rewrite를 그대로 생성한다.|DP 분석 결과를 통한 부사절 제시|
|gold_rewrite|아이들용 부대시설도 많기 때문에 많은 분들이 가족끼리 여행을 오시더라구요.|관형절, 명사절 활용하여 다시 쓴 문장|\


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
