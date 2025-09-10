#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoNLL-U 파일을 JSONL 형식으로 변환하는 스크립트
입력: ko_gsd-ud-train.conllu
출력: sample_data.jsonl 형식의 JSONL 파일
"""

import json
import re
import random
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def parse_conllu_file(file_path):
    """
    CoNLL-U 파일을 파싱하여 문장별로 분리
    """
    sentences = []
    current_sentence = {
        'id': None,
        'text': None,
        'tokens': []
    }
    
    # 파일의 총 라인 수를 먼저 계산
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="파일 파싱 중"):
            line = line.strip()
            
            # 빈 줄: 문장 구분자
            if not line:
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                    current_sentence = {
                        'id': None,
                        'text': None,
                        'tokens': []
                    }
                continue
            
            # 주석 줄
            if line.startswith('#'):
                if line.startswith('# sent_id ='):
                    current_sentence['id'] = line.split('=', 1)[1].strip()
                elif line.startswith('# text ='):
                    current_sentence['text'] = line.split('=', 1)[1].strip()
                continue
            
            # 토큰 줄 파싱
            parts = line.split('\t')
            if len(parts) >= 8:
                token = {
                    'id': parts[0],
                    'form': parts[1],
                    'lemma': parts[2],
                    'upos': parts[3],
                    'xpos': parts[4],
                    'feats': parts[5],
                    'head': parts[6],
                    'deprel': parts[7],
                    'deps': parts[8] if len(parts) > 8 else '_',
                    'misc': parts[9] if len(parts) > 9 else '_'
                }
                current_sentence['tokens'].append(token)
    
    # 마지막 문장 추가
    if current_sentence['tokens']:
        sentences.append(current_sentence)
    
    return sentences

def check_dependency_tag_exists(gold_dep, target_tag):
    """
    gold_dep에서 특정 의존관계 태그가 존재하는지 확인
    
    Args:
        gold_dep: CoNLL-U 형식의 의존관계 분석 결과
        target_tag: 찾을 태그 (예: 'csubj', 'ccomp', 'acl:relcl', 'advcl')
    
    Returns:
        bool: 태그가 존재하면 True, 없으면 False
    """
    if not gold_dep:
        return False
    
    # 디버깅을 위한 로그 추가
    #print(f"Looking for tag: {target_tag}")
    #print(f"Gold_dep content:\n{gold_dep}")
    
    # gold_dep의 각 줄을 파싱하여 deprel 필드 확인
    for line_num, line in enumerate(gold_dep.strip().split('\n'), 1):
        if line.strip():
            parts = line.split('\t')
            #print(f"Line {line_num}: {parts}")
            
            if len(parts) >= 8:  # CoNLL-U 형식: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                deprel = parts[7]  # DEPREL은 8번째 필드 (인덱스 7)
                #print(f"  DEPREL: {deprel}")
                if deprel == target_tag:
                    #print(f"  Found tag {target_tag}!")
                    return True
            else:
                print(f"  Warning: Line has only {len(parts)} fields, expected at least 8")
    
    #print(f"  Tag {target_tag} not found")
    return False

def get_appropriate_rewrite_instruction(gold_dep):
    """
    gold_dep의 의존관계를 분석하여 적절한 rewrite_instruction을 선택
    
    Args:
        gold_dep: CoNLL-U 형식의 의존관계 분석 결과
    
    Returns:
        tuple: (instruction, tag_found) - 선택된 지시사항과 찾은 태그, 태그가 없으면 (None, None)
    """
    # 태그별 지시사항 매핑
    tag_instructions = {
        'csubj': "gold_dep에 명사절(csubj)이 있을 때 명사절의 명사화 표지를 제거하여 문장을 재작성하라.",
        'ccomp': "gold_dep에 인용절(ccomp)이 있을 때 인용절에 불필요한 인용 표지를 추가하여 문장을 재작성하라.",
        'acl:relcl': "gold_dep에 관형절(acl:relcl)이 있을 때 관형절에 존재하지 않는 수식어를 삽입하여 문장을 재작성하라.",
        'advcl': "gold_dep에 부사절(advcl)이 있을 때 부사절이 원래 의존하던 주절 서술어가 아닌 다른 서술어에 걸리도록 문장을 재작성하라."
    }
    
    #print(f"Analyzing gold_dep for tags: {list(tag_instructions.keys())}")
    
    # gold_dep에서 찾을 수 있는 태그들 확인
    found_tags = []
    for tag in tag_instructions.keys():
        if check_dependency_tag_exists(gold_dep, tag):
            found_tags.append(tag)
    
    #print(f"Found tags: {found_tags}")
    
    if found_tags:
        # 찾은 태그 중에서 무작위로 하나 선택
        selected_tag = random.choice(found_tags)
        return tag_instructions[selected_tag], selected_tag
    else:
        # 적절한 태그가 없으면 None 반환
        return None, None

def generate_gold_rewrite_with_openai(text, instruction, gold_dep, api_key=None, model="gpt-4o"):
    """
    OpenAI API를 사용하여 gold_rewrite 생성 (gold_dep 정보 포함)
    """
    try:
        client = OpenAI(api_key=api_key)
        
        system_prompt = """당신은 한국어 문장 재작성 전문가입니다. 
주어진 지시사항에 따라 문장을 재작성하세요. 
재작성된 문장은 자연스럽고 문법적으로 올바르며, 원래 의미를 유지해야 합니다.
답변에는 재작성된 문장만 포함하세요."""

        user_prompt = f"""원본 문장: {text}

의존관계 분석 결과:
{gold_dep}

지시사항: {instruction}

재작성된 문장:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"OpenAI API 오류: {e}")
        # API 오류 시 원본 텍스트 반환
        return text

def convert_to_jsonl_format(sentences, max_sentences=100, start_index=0, use_openai=False, api_key=None, model="gpt-4o-mini"):
    """
    문장들을 JSONL 형식으로 변환
    """
    jsonl_data = []
    skipped_count = 0
    
    # 시작 인덱스부터 처리할 문장들 선택
    sentences_to_process = sentences[start_index:start_index + max_sentences]
    
    # tqdm을 사용하여 진행률 표시
    for sentence in tqdm(sentences_to_process, desc=f"문장 처리 중 ({start_index+1}~{start_index+len(sentences_to_process)})"):
        # 문장 ID 생성
        sent_id = sentence['id'] if sentence['id'] else f"s{start_index + len(jsonl_data) + 1}"
        
        # 원본 텍스트
        text = sentence['text'] if sentence['text'] else ' '.join([token['form'] for token in sentence['tokens']])
        
        # CoNLL-U 형식의 gold_dep 생성 (올바른 필드 순서)
        gold_dep_lines = []
        for token in sentence['tokens']:
            # CoNLL-U 형식: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            line = f"{token['id']}\t{token['form']}\t{token['lemma']}\t{token['upos']}\t{token['xpos']}\t{token['feats']}\t{token['head']}\t{token['deprel']}\t{token['deps']}\t{token['misc']}"
            gold_dep_lines.append(line)
        
        gold_dep = '\n'.join(gold_dep_lines)
        
        #print(f"\nProcessing sentence: {sent_id}")
        #print(f"Text: {text}")
        
        # gold_dep를 분석하여 적절한 rewrite_instruction 선택
        rewrite_instruction, found_tag = get_appropriate_rewrite_instruction(gold_dep)
        
        # 적절한 태그가 없으면 이 문장을 건너뛰기
        if rewrite_instruction is None:
            #print(f"Skipping sentence {sent_id} - no appropriate tags found")
            skipped_count += 1
            continue
        
        #print(f"Selected tag: {found_tag}")
        #print(f"Instruction: {rewrite_instruction}")
        
        # gold_rewrite 생성
        if use_openai and api_key:
            gold_rewrite = generate_gold_rewrite_with_openai(text, rewrite_instruction, gold_dep, api_key, model)
            # API 호출 간격 조절 (rate limit 방지)
            time.sleep(1)
        else:
            # OpenAI를 사용하지 않는 경우, found_tag가 있을 때만 재작성 시도
            if found_tag:
                # 간단한 재작성 로직 (실제로는 더 복잡한 로직이 필요할 수 있음)
                gold_rewrite = f"[재작성됨 - {found_tag} 태그 발견] {text}"
            else:
                gold_rewrite = text
        
        # JSONL 항목 생성
        jsonl_item = {
            "id": sent_id,
            "text": text,
            "gold_dep": gold_dep,
            "rewrite_instruction": rewrite_instruction,
            "gold_rewrite": gold_rewrite
        }
        
        jsonl_data.append(jsonl_item)
    
    # 건너뛴 문장 수 출력
    if skipped_count > 0:
        print(f"건너뛴 문장 수: {skipped_count}개 (적절한 의존관계 태그가 없음)")
    
    return jsonl_data

def save_jsonl(data, output_path, append=False):
    """
    JSONL 데이터를 파일로 저장
    """
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for item in tqdm(data, desc="파일 저장 중"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 입력 파일 경로
    input_file = "4. ud/ko_gsd-ud-train.conllu" #
    output_file = "llm_clause_experiment_py/ko_gsd-ud-train.jsonl"
    
    # OpenAI API 설정
    use_openai = input("OpenAI API를 사용하여 gold_rewrite를 생성하시겠습니까? (y/n): ").lower().startswith('y')
    api_key = None
    model = "gpt-4o-mini"
    
    if use_openai:
        api_key = input("OpenAI API 키를 입력하세요: ").strip() #my key: "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"
        if not api_key:
            print("API 키가 입력되지 않았습니다. 원본 텍스트를 사용합니다.")
            use_openai = False
        else:
            model_choice = input("사용할 모델을 선택하세요 (1: gpt-4o-mini, 2: gpt-4o, 3: gpt-3.5-turbo): ").strip()
            if model_choice == "2":
                model = "gpt-4o"
            elif model_choice == "3":
                model = "gpt-3.5-turbo"
    
    print(f"CoNLL-U 파일을 파싱 중: {input_file}")
    
    # CoNLL-U 파일 파싱
    sentences = parse_conllu_file(input_file)
    print(f"총 {len(sentences)}개의 문장을 찾았습니다.")
    
    # 시작 문장 번호 설정 (1부터 시작하는 번호)
    start_sentence = input("시작 문장 번호를 입력하세요 (1부터 시작, 기본값: 1): ").strip()
    start_index = int(start_sentence) - 1 if start_sentence.isdigit() else 0
    
    # 처리할 문장 수 설정
    max_sentences = input("처리할 문장 수를 입력하세요 (기본값: 10): ").strip()
    max_sentences = int(max_sentences) if max_sentences.isdigit() else 10
    
    # 기존 파일에 추가할지 새로 생성할지 선택
    append_mode = False
    if start_index > 0:
        append_choice = input(f"기존 파일에 {start_index+1}번 문장부터 추가하시겠습니까? (y/n): ").lower().startswith('y')
        append_mode = append_choice
    
    # 유효성 검사
    if start_index >= len(sentences):
        print(f"오류: 시작 문장 번호({start_index+1})가 총 문장 수({len(sentences)})를 초과합니다.")
        return
    
    if start_index + max_sentences > len(sentences):
        max_sentences = len(sentences) - start_index
        print(f"경고: 요청한 문장 수가 남은 문장 수를 초과하여 {max_sentences}개로 조정됩니다.")
    
    print(f"{start_index+1}번 문장부터 {start_index+max_sentences}번 문장까지 처리합니다.")
    
    # JSONL 형식으로 변환
    jsonl_data = convert_to_jsonl_format(sentences, max_sentences=max_sentences, start_index=start_index,
                                       use_openai=use_openai, api_key=api_key, model=model)
    print(f"{len(jsonl_data)}개의 문장을 JSONL 형식으로 변환했습니다.")
    
    # 파일 저장
    save_jsonl(jsonl_data, output_file, append=append_mode)
    mode_str = "추가" if append_mode else "새로 생성"
    print(f"JSONL 파일이 {mode_str}되었습니다: {output_file}")
    
    # 샘플 출력
    #if jsonl_data:
    #    print("\n첫 번째 항목 샘플:")
    #    print(json.dumps(jsonl_data[0], ensure_ascii=False, indent=2))
    # apikey: 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
if __name__ == "__main__":
    main() 