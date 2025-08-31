# -*- coding: utf-8 -*-
"""
JSONL과 XLSX 간 변환 유틸리티
- jsonl_to_xlsx: JSONL 파일을 XLSX로 변환
- xlsx_to_jsonl: XLSX 파일을 JSONL로 변환
"""
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def jsonl_to_xlsx(jsonl_file_path: str, 
                  xlsx_file_path: str, 
                  columns: Optional[List[str]] = None) -> None:
    """
    JSONL 파일을 XLSX 파일로 변환합니다.
    
    Args:
        jsonl_file_path: 입력 JSONL 파일 경로
        xlsx_file_path: 출력 XLSX 파일 경로
        columns: 포함할 컬럼 리스트 (None이면 모든 컬럼 포함)
    """
    try:
        logger.info(f"Reading JSONL file: {jsonl_file_path}")
        
        # JSONL 파일 읽기
        data = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 빈 줄 건너뛰기
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
        
        if not data:
            logger.error("No valid data found in JSONL file")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(data)
        
        # 컬럼 필터링
        if columns:
            # 존재하는 컬럼만 필터링
            available_columns = [col for col in columns if col in df.columns]
            missing_columns = [col for col in columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            df = df[available_columns]
        
        # XLSX 파일로 저장
        logger.info(f"Saving to XLSX file: {xlsx_file_path}")
        df.to_excel(xlsx_file_path, index=False, engine='openpyxl')
        
        logger.info(f"Successfully converted {len(df)} rows to XLSX")
        logger.info(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        logger.error(f"Error converting JSONL to XLSX: {e}")
        raise


def xlsx_to_jsonl(xlsx_file_path: str, 
                  jsonl_file_path: str, 
                  sheet_name: Optional[str] = None) -> None:
    """
    XLSX 파일을 JSONL 파일로 변환합니다.
    
    Args:
        xlsx_file_path: 입력 XLSX 파일 경로
        jsonl_file_path: 출력 JSONL 파일 경로
        sheet_name: 시트 이름 (None이면 첫 번째 시트)
    """
    try:
        logger.info(f"Reading XLSX file: {xlsx_file_path}")
        
        # XLSX 파일 읽기
        df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, engine='openpyxl')
        
        # sheet_name이 None이고 딕셔너리가 반환된 경우 첫 번째 시트 사용
        if isinstance(df, dict):
            if sheet_name is None:
                # 첫 번째 시트 사용
                first_sheet = list(df.keys())[0]
                logger.info(f"Multiple sheets found. Using first sheet: {first_sheet}")
                df = df[first_sheet]
            else:
                raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {list(df.keys())}")
        
        # NaN 값을 None으로 변환
        df = df.where(pd.notnull(df), None)
        
        # JSONL 파일로 저장
        logger.info(f"Saving to JSONL file: {jsonl_file_path}")
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # NaN 값을 None으로 변환하고 JSON으로 직렬화
                row_dict = row.to_dict()
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                
                json_line = json.dumps(row_dict, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Successfully converted {len(df)} rows to JSONL")
        logger.info(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        logger.error(f"Error converting XLSX to JSONL: {e}")
        raise


# 사용 예시
if __name__ == "__main__":
    import os
    
    # 스크립트가 있는 디렉토리로 작업 디렉토리 변경
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # JSONL을 XLSX로 변환
  #  jsonl_to_xlsx(
  #      jsonl_file_path="ko_gsd_train.jsonl",
  #      xlsx_file_path="ko_gsd_train.xlsx",
  #      columns=["id", "text", "gold_dep", "rewrite_instruction", "gold_rewrite"]
  #  )
    
    # XLSX를 JSONL로 변환
    xlsx_to_jsonl(
        xlsx_file_path="ko_gsd_train_final.xlsx",
        jsonl_file_path="ko_gsd_train_final.jsonl"
    )
