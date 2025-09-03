# -*- coding: utf-8 -*-
"""
Enhanced Visualization for Parser Evaluation Results
- Stanza vs Gold_dep 비교 그래프 (UAS, LAS)
- LLM 구문분석 성능 비교 (GPT-4o, GPT-5, Claude)
- LLM 재작성 성능 비교 (GPT-4o, GPT-5, Claude)
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Font settings for English
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_enhanced_results(jsonl_path: str) -> pd.DataFrame:
    """향상된 평가 결과를 로드합니다."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return pd.DataFrame(data)

def create_stanza_parser_visualizations(df: pd.DataFrame, output_dir: str):
    """Stanza 구문분석기 시각화를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Stanza 데이터 추출
    stanza_data = []
    
    for _, row in df.iterrows():
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            stanza_data.append({
                "id": row["id"],
                "text": row["text"],
                "UAS": metrics.get("UAS", 0.0),
                "LAS": metrics.get("LAS", 0.0),
                "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
            })
    
    # 데이터프레임 생성
    stanza_df = pd.DataFrame(stanza_data)
    
    if stanza_df.empty:
        print("No Stanza parser data available")
        return
    
    # 1. UAS 분포 히스토그램
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(stanza_df['UAS'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Stanza UAS Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('UAS Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. LAS 분포 히스토그램
    plt.subplot(2, 3, 2)
    plt.hist(stanza_df['LAS'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Stanza LAS Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('LAS Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 3. UAS vs LAS 스캐터 플롯
    plt.subplot(2, 3, 3)
    plt.scatter(stanza_df['UAS'], stanza_df['LAS'], alpha=0.6, s=50, color='green')
    plt.xlabel('UAS Score')
    plt.ylabel('LAS Score')
    plt.title('UAS vs LAS Scatter Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. ClauseSpanF1 분포 히스토그램
    plt.subplot(2, 3, 4)
    plt.hist(stanza_df['ClauseSpanF1'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('ClauseSpanF1 Score')
    plt.ylabel('Frequency')
    plt.title('Stanza ClauseSpanF1 Score Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. 성능 통계 요약
    plt.subplot(2, 3, 5)
    plt.axis('off')
    stats_text = f"""
    Stanza Parser Performance Statistics:
    
    Total Samples: {len(stanza_df)}
    Mean UAS: {stanza_df['UAS'].mean():.4f}
    Std UAS: {stanza_df['UAS'].std():.4f}
    Min UAS: {stanza_df['UAS'].min():.4f}
    Max UAS: {stanza_df['UAS'].max():.4f}
    
    Mean LAS: {stanza_df['LAS'].mean():.4f}
    Std LAS: {stanza_df['LAS'].std():.4f}
    Min LAS: {stanza_df['LAS'].min():.4f}
    Max LAS: {stanza_df['LAS'].max():.4f}
    
    Mean ClauseSpanF1: {stanza_df['ClauseSpanF1'].mean():.4f}
    Std ClauseSpanF1: {stanza_df['ClauseSpanF1'].std():.4f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', transform=plt.gca().transAxes)
    
    # 6. 메트릭 박스플롯
    plt.subplot(2, 3, 6)
    metrics_data = []
    for metric in ['UAS', 'LAS', 'ClauseSpanF1']:
        for value in stanza_df[metric]:
            metrics_data.append({'Metric': metric, 'Score': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    sns.boxplot(data=metrics_df, x='Metric', y='Score', color='lightgreen')
    plt.title('Stanza Metrics Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "stanza_parser_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 상세 통계 테이블
    comparison_stats = stanza_df[['UAS', 'LAS', 'ClauseSpanF1']].agg(['mean', 'std', 'min', 'max']).round(4)
    comparison_stats.to_csv(output_path / "stanza_parser_statistics.csv")
    
    print(f"Stanza parser visualizations saved to {output_path}")

def create_llm_dependency_visualizations(df: pd.DataFrame, output_dir: str):
    """LLM 구문분석 성능 시각화를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # LLM 구문분석 데이터 추출
    llm_dep_data = []
    
    for _, row in df.iterrows():
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    llm_dep_data.append({
                        "id": row["id"],
                        "model": model,
                        "UAS": metrics.get("UAS", 0.0),
                        "LAS": metrics.get("LAS", 0.0),
                        "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
                    })
    
    if not llm_dep_data:
        print("No LLM dependency parsing data available")
        return
    
    llm_dep_df = pd.DataFrame(llm_dep_data)
    
    # 1. 모델별 성능 비교
    plt.figure(figsize=(15, 10))
    
    # UAS 비교
    plt.subplot(2, 3, 1)
    models = llm_dep_df['model'].unique()
    uas_means = [llm_dep_df[llm_dep_df['model'] == model]['UAS'].mean() for model in models]
    plt.bar(models, uas_means, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.title('Average UAS by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Average UAS')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # LAS 비교
    plt.subplot(2, 3, 2)
    las_means = [llm_dep_df[llm_dep_df['model'] == model]['LAS'].mean() for model in models]
    plt.bar(models, las_means, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.title('Average LAS by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Average LAS')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # ClauseSpanF1 비교
    plt.subplot(2, 3, 3)
    clause_means = [llm_dep_df[llm_dep_df['model'] == model]['ClauseSpanF1'].mean() for model in models]
    plt.bar(models, clause_means, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.title('Average ClauseSpanF1 by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Average ClauseSpanF1')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. 모델별 메트릭 분포
    plt.subplot(2, 3, 4)
    sns.boxplot(data=llm_dep_df, x='model', y='UAS', palette=['blue', 'green', 'orange'])
    plt.title('UAS Distribution by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    sns.boxplot(data=llm_dep_df, x='model', y='LAS', palette=['blue', 'green', 'orange'])
    plt.title('LAS Distribution by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    sns.boxplot(data=llm_dep_df, x='model', y='ClauseSpanF1', palette=['blue', 'green', 'orange'])
    plt.title('ClauseSpanF1 Distribution by Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "llm_dependency_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 상세 통계 테이블
    model_stats = llm_dep_df.groupby('model')[['UAS', 'LAS', 'ClauseSpanF1']].agg(['mean', 'std', 'min', 'max']).round(4)
    model_stats.to_csv(output_path / "llm_dependency_statistics.csv")
    
    print(f"LLM dependency parsing visualizations saved to {output_path}")

def create_llm_rewrite_visualizations(df: pd.DataFrame, output_dir: str):
    """LLM 재작성 성능 시각화를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # LLM 재작성 데이터 추출
    llm_rewrite_data = []
    
    for _, row in df.iterrows():
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    llm_rewrite_data.append({
                        "id": row["id"],
                        "model": model,
                        "best_f1_score": results.get("best_f1_score", 0.0),
                        "avg_f1_score": results.get("avg_f1_score", 0.0),
                        "best_f1_score_without_advcl": results.get("best_f1_score_without_advcl", 0.0),
                        "avg_f1_score_without_advcl": results.get("avg_f1_score_without_advcl", 0.0),
                        "best_rewritten_text": results.get("best_rewritten_text", ""),
                        "best_rewritten_text_without_advcl": results.get("best_rewritten_text_without_advcl", ""),
                        "num_clauses": len(results.get("clauses", [])),
                        "num_rewrites": len(results.get("rewrite_results", [])),
                        "num_rewrites_without_advcl": len(results.get("rewrite_results_without_advcl", []))
                    })
    
    if not llm_rewrite_data:
        print("No LLM rewrite data available")
        return
    
    llm_rewrite_df = pd.DataFrame(llm_rewrite_data)
    
    # 1. 모델별 F1 점수 비교 (advcl 정보 포함 vs 제외)
    plt.figure(figsize=(15, 10))
    
    # Best F1 Score 비교 (advcl 정보 포함 vs 제외)
    plt.subplot(2, 3, 1)
    models = llm_rewrite_df['model'].unique()
    
    # advcl 정보 포함한 결과
    best_f1_means_with_advcl = [llm_rewrite_df[llm_rewrite_df['model'] == model]['best_f1_score'].mean() for model in models]
    # advcl 정보 제외한 결과
    best_f1_means_without_advcl = [llm_rewrite_df[llm_rewrite_df['model'] == model]['best_f1_score_without_advcl'].mean() for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, best_f1_means_with_advcl, width, label='With Advcl Info', color='blue', alpha=0.7)
    plt.bar(x + width/2, best_f1_means_without_advcl, width, label='Without Advcl Info', color='orange', alpha=0.7)
    
    plt.title('Average Best F1 Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average Best F1 Score')
    plt.xlabel('Model')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 모델별 F1 점수 분포 (박스플롯)
    plt.subplot(2, 3, 2)
    
    # advcl 정보 포함한 결과
    with_advcl_data = []
    without_advcl_data = []
    labels = []
    
    for model in models:
        model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
        with_advcl_data.extend(model_data['best_f1_score'].values)
        without_advcl_data.extend(model_data['best_f1_score_without_advcl'].values)
        labels.extend([f'{model}\n(With Advcl)'] * len(model_data))
        labels.extend([f'{model}\n(Without Advcl)'] * len(model_data))
    
    all_data = with_advcl_data + without_advcl_data
    colors = ['blue'] * len(with_advcl_data) + ['orange'] * len(without_advcl_data)
    
    box_data = [with_advcl_data, without_advcl_data]
    plt.boxplot(box_data, labels=['With Advcl Info', 'Without Advcl Info'], patch_artist=True)
    plt.title('F1 Score Distribution Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Best F1 Score')
    plt.grid(True, alpha=0.3)
    
    # 3. 성능 향상도 분석
    plt.subplot(2, 3, 3)
    
    improvement_data = []
    for model in models:
        model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
        for _, row in model_data.iterrows():
            if row['best_f1_score_without_advcl'] > 0:  # 0으로 나누기 방지
                improvement = (row['best_f1_score'] - row['best_f1_score_without_advcl']) / row['best_f1_score_without_advcl'] * 100
                improvement_data.append(improvement)
    
    if improvement_data:
        plt.hist(improvement_data, bins=10, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(improvement_data), color='red', linestyle='--', label=f'Mean: {np.mean(improvement_data):.1f}%')
        plt.title('Performance Improvement with Advcl Info', fontsize=14, fontweight='bold')
        plt.xlabel('Improvement (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. 모델별 성능 향상도
    plt.subplot(2, 3, 4)
    
    model_improvements = []
    for model in models:
        model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
        improvements = []
        for _, row in model_data.iterrows():
            if row['best_f1_score_without_advcl'] > 0:
                improvement = (row['best_f1_score'] - row['best_f1_score_without_advcl']) / row['best_f1_score_without_advcl'] * 100
                improvements.append(improvement)
        
        if improvements:
            model_improvements.append(np.mean(improvements))
        else:
            model_improvements.append(0)
    
    plt.bar(models, model_improvements, color=['blue', 'green', 'orange'], alpha=0.7)
    plt.title('Average Performance Improvement by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Average Improvement (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. 성능 등급 분포 비교
    plt.subplot(2, 3, 5)
    
    def classify_f1_score(score):
        if score >= 0.8:
            return 'Excellent (≥0.8)'
        elif score >= 0.6:
            return 'Good (0.6-0.8)'
        elif score >= 0.4:
            return 'Fair (0.4-0.6)'
        else:
            return 'Poor (<0.4)'
    
    llm_rewrite_df['performance_grade_with_advcl'] = llm_rewrite_df['best_f1_score'].apply(classify_f1_score)
    llm_rewrite_df['performance_grade_without_advcl'] = llm_rewrite_df['best_f1_score_without_advcl'].apply(classify_f1_score)
    
    # advcl 정보 포함한 성능 등급
    grade_with_advcl = pd.crosstab(llm_rewrite_df['model'], llm_rewrite_df['performance_grade_with_advcl'])
    grade_without_advcl = pd.crosstab(llm_rewrite_df['model'], llm_rewrite_df['performance_grade_without_advcl'])
    
    colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
    
    # advcl 정보 포함한 결과만 표시
    grade_with_advcl.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors[:len(grade_with_advcl.columns)])
    plt.title('Performance Grade with Advcl Info', fontsize=14, fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Performance Grade')
    plt.grid(True, alpha=0.3)
    
    # 6. 성능 통계 요약
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    stats_text = f"""
    LLM Rewrite Performance Comparison:
    
    Total Samples: {len(llm_rewrite_df)}
    
    By Model:
    """
    for model in models:
        model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
        if not model_data.empty:
            stats_text += f"""
    {model}:
    - With Advcl: {model_data['best_f1_score'].mean():.4f} ± {model_data['best_f1_score'].std():.4f}
    - Without Advcl: {model_data['best_f1_score_without_advcl'].mean():.4f} ± {model_data['best_f1_score_without_advcl'].std():.4f}
    - Improvement: {((model_data['best_f1_score'].mean() - model_data['best_f1_score_without_advcl'].mean()) / model_data['best_f1_score_without_advcl'].mean() * 100):.1f}%
    """
    
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path / "llm_rewrite_comparison_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 상세 통계 저장
    comparison_stats = []
    for model in models:
        model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
        if not model_data.empty:
            comparison_stats.append({
                'Model': model,
                'Count': len(model_data),
                'Mean_F1_With_Advcl': model_data['best_f1_score'].mean(),
                'Std_F1_With_Advcl': model_data['best_f1_score'].std(),
                'Mean_F1_Without_Advcl': model_data['best_f1_score_without_advcl'].mean(),
                'Std_F1_Without_Advcl': model_data['best_f1_score_without_advcl'].std(),
                'Improvement_Percent': ((model_data['best_f1_score'].mean() - model_data['best_f1_score_without_advcl'].mean()) / model_data['best_f1_score_without_advcl'].mean() * 100) if model_data['best_f1_score_without_advcl'].mean() > 0 else 0
            })
    
    if comparison_stats:
        comparison_df = pd.DataFrame(comparison_stats)
        comparison_df.to_csv(output_path / "llm_rewrite_comparison_statistics.csv", index=False)
    
    # 8. 최고 성능 예제들 저장
    best_examples = llm_rewrite_df.nlargest(10, 'best_f1_score')[['id', 'model', 'best_f1_score', 'best_f1_score_without_advcl', 'best_rewritten_text', 'best_rewritten_text_without_advcl']]
    best_examples.to_csv(output_path / "best_llm_rewrite_comparison_examples.csv", index=False, encoding='utf-8')
    
    print(f"LLM rewrite comparison visualizations saved to {output_path}")

def create_comprehensive_comparison(df: pd.DataFrame, output_dir: str):
    """종합 비교 시각화를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 모델의 성능 데이터 수집
    all_models_data = []
    
    # Stanza 데이터
    for _, row in df.iterrows():
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            all_models_data.append({
                "id": row["id"],
                "model": "Stanza",
                "UAS": metrics.get("UAS", 0.0),
                "LAS": metrics.get("LAS", 0.0),
                "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0),
                "task": "dependency_parsing"
            })
    
    # LLM 구문분석 데이터
    for _, row in df.iterrows():
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    all_models_data.append({
                        "id": row["id"],
                        "model": model,
                        "UAS": metrics.get("UAS", 0.0),
                        "LAS": metrics.get("LAS", 0.0),
                        "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0),
                        "task": "dependency_parsing"
                    })
    
    # LLM 재작성 데이터
    for _, row in df.iterrows():
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    all_models_data.append({
                        "id": row["id"],
                        "model": model,
                        "F1_Score": results.get("best_f1_score", 0.0),
                        "task": "text_rewriting"
                    })
    
    if not all_models_data:
        print("No comprehensive comparison data available")
        return
    
    all_models_df = pd.DataFrame(all_models_data)
    
    # 1. 구문분석 성능 비교
    dep_data = all_models_df[all_models_df['task'] == 'dependency_parsing']
    if not dep_data.empty:
        plt.figure(figsize=(15, 5))
        
        # UAS 비교
        plt.subplot(1, 3, 1)
        sns.boxplot(data=dep_data, x='model', y='UAS')
        plt.title('UAS Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # LAS 비교
        plt.subplot(1, 3, 2)
        sns.boxplot(data=dep_data, x='model', y='LAS')
        plt.title('LAS Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # ClauseSpanF1 비교
        plt.subplot(1, 3, 3)
        sns.boxplot(data=dep_data, x='model', y='ClauseSpanF1')
        plt.title('ClauseSpanF1 Comparison Across Models', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "comprehensive_dependency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 재작성 성능 비교
    rewrite_data = all_models_df[all_models_df['task'] == 'text_rewriting']
    if not rewrite_data.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=rewrite_data, x='model', y='F1_Score')
        plt.title('F1 Score Comparison for Text Rewriting', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "comprehensive_rewrite_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 종합 통계 테이블
    if not dep_data.empty:
        dep_stats = dep_data.groupby('model')[['UAS', 'LAS', 'ClauseSpanF1']].agg(['mean', 'std']).round(4)
        dep_stats.to_csv(output_path / "comprehensive_dependency_statistics.csv")
    
    if not rewrite_data.empty:
        rewrite_stats = rewrite_data.groupby('model')['F1_Score'].agg(['count', 'mean', 'std']).round(4)
        rewrite_stats.to_csv(output_path / "comprehensive_rewrite_statistics.csv")
    
    print(f"Comprehensive comparison visualizations saved to {output_path}")

def create_metrics_distribution_visualizations(df: pd.DataFrame, output_dir: str):
    """메트릭 분포 히스토그램을 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 모델의 메트릭 데이터 수집
    all_metrics = []
    
    # Stanza 메트릭
    for _, row in df.iterrows():
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            all_metrics.append({
                "model": "Stanza",
                "UAS": metrics.get("UAS", 0.0),
                "LAS": metrics.get("LAS", 0.0),
                "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
            })
    
    # LLM 구문분석 메트릭
    for _, row in df.iterrows():
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    all_metrics.append({
                        "model": model,
                        "UAS": metrics.get("UAS", 0.0),
                        "LAS": metrics.get("LAS", 0.0),
                        "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
                    })
    
    # LLM 재작성 F1 점수
    for _, row in df.iterrows():
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    all_metrics.append({
                        "model": model,
                        "F1_Score": results.get("best_f1_score", 0.0)
                    })
    
    if not all_metrics:
        print("No metrics data available for distribution analysis")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # 1. 메트릭 분포 히스토그램
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Evaluation Metrics Distribution by Model', fontsize=16, fontweight='bold')
    
    # UAS 분포
    if 'UAS' in metrics_df.columns:
        for i, model in enumerate(metrics_df['model'].unique()):
            model_data = metrics_df[metrics_df['model'] == model]['UAS'].dropna()
            if not model_data.empty:
                axes[0, 0].hist(model_data, bins=20, alpha=0.7, label=model)
        axes[0, 0].set_title('UAS Distribution')
        axes[0, 0].set_xlabel('UAS')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # LAS 분포
    if 'LAS' in metrics_df.columns:
        for i, model in enumerate(metrics_df['model'].unique()):
            model_data = metrics_df[metrics_df['model'] == model]['LAS'].dropna()
            if not model_data.empty:
                axes[0, 1].hist(model_data, bins=20, alpha=0.7, label=model)
        axes[0, 1].set_title('LAS Distribution')
        axes[0, 1].set_xlabel('LAS')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # ClauseSpanF1 분포
    if 'ClauseSpanF1' in metrics_df.columns:
        for i, model in enumerate(metrics_df['model'].unique()):
            model_data = metrics_df[metrics_df['model'] == model]['ClauseSpanF1'].dropna()
            if not model_data.empty:
                axes[0, 2].hist(model_data, bins=20, alpha=0.7, label=model)
        axes[0, 2].set_title('ClauseSpanF1 Distribution')
        axes[0, 2].set_xlabel('ClauseSpanF1')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # F1 점수 분포
    if 'F1_Score' in metrics_df.columns:
        for i, model in enumerate(metrics_df['model'].unique()):
            model_data = metrics_df[metrics_df['model'] == model]['F1_Score'].dropna()
            if not model_data.empty:
                axes[1, 0].hist(model_data, bins=20, alpha=0.7, label=model)
        axes[1, 0].set_title('F1 Score Distribution')
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 나머지 subplot 숨기기
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "metrics_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics distribution visualizations saved to {output_path}")

def create_metrics_boxplot_visualizations(df: pd.DataFrame, output_dir: str):
    """메트릭 박스플롯을 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 모델의 메트릭 데이터 수집
    all_metrics = []
    
    # Stanza 메트릭
    for _, row in df.iterrows():
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            all_metrics.append({
                "model": "Stanza",
                "UAS": metrics.get("UAS", 0.0),
                "LAS": metrics.get("LAS", 0.0),
                "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
            })
    
    # LLM 구문분석 메트릭
    for _, row in df.iterrows():
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    all_metrics.append({
                        "model": model,
                        "UAS": metrics.get("UAS", 0.0),
                        "LAS": metrics.get("LAS", 0.0),
                        "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
                    })
    
    # LLM 재작성 F1 점수
    for _, row in df.iterrows():
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    all_metrics.append({
                        "model": model,
                        "F1_Score": results.get("best_f1_score", 0.0)
                    })
    
    if not all_metrics:
        print("No metrics data available for boxplot analysis")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # 1. 구문분석 메트릭 박스플롯
    dep_metrics = ['UAS', 'LAS', 'ClauseSpanF1']
    available_dep_metrics = [m for m in dep_metrics if m in metrics_df.columns]
    
    if available_dep_metrics:
        fig, axes = plt.subplots(1, len(available_dep_metrics), figsize=(5*len(available_dep_metrics), 6))
        if len(available_dep_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_dep_metrics):
            sns.boxplot(data=metrics_df, x='model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} Distribution by Model')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "dependency_metrics_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 재작성 F1 점수 박스플롯
    if 'F1_Score' in metrics_df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df, x='model', y='F1_Score')
        plt.title('F1 Score Distribution by Model', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "rewrite_f1_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Metrics boxplot visualizations saved to {output_path}")

def create_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """상관관계 히트맵을 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 메트릭을 하나의 데이터프레임으로 통합
    all_metrics = []
    
    for _, row in df.iterrows():
        metrics_row = {"id": row["id"]}
        
        # Stanza 메트릭
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            metrics_row.update({
                "stanza_UAS": metrics.get("UAS", 0.0),
                "stanza_LAS": metrics.get("LAS", 0.0),
                "stanza_ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
            })
        
        # LLM 구문분석 메트릭
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    metrics_row.update({
                        f"{model}_UAS": metrics.get("UAS", 0.0),
                        f"{model}_LAS": metrics.get("LAS", 0.0),
                        f"{model}_ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0)
                    })
        
        # LLM 재작성 F1 점수
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    metrics_row[f"{model}_F1"] = results.get("best_f1_score", 0.0)
        
        all_metrics.append(metrics_row)
    
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # 숫자 컬럼만 선택
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'id']
        
        if len(numeric_cols) > 1:
            correlation_matrix = metrics_df[numeric_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
            plt.title('Correlation Between All Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 상관관계 통계 저장
            correlation_matrix.to_csv(output_path / "correlation_statistics.csv")
            
            print(f"Correlation heatmap saved to {output_path}")

def create_comprehensive_report(df: pd.DataFrame, output_dir: str):
    """종합 리포트를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Enhanced Parser Evaluation Comprehensive Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Total samples: {len(df)}")
    report_lines.append("")
    
    # Stanza 성능 요약
    stanza_count = sum(1 for _, row in df.iterrows() if row.get("stanza_results"))
    report_lines.append("Stanza Parser Performance:")
    report_lines.append("-" * 40)
    report_lines.append(f"Available samples: {stanza_count}")
    
    if stanza_count > 0:
        stanza_metrics = []
        for _, row in df.iterrows():
            if row.get("stanza_results") and row["stanza_results"].get("metrics"):
                stanza_metrics.append(row["stanza_results"]["metrics"])
        
        if stanza_metrics:
            stanza_df = pd.DataFrame(stanza_metrics)
            report_lines.append(f"Mean UAS: {stanza_df['UAS'].mean():.4f}")
            report_lines.append(f"Mean LAS: {stanza_df['LAS'].mean():.4f}")
            report_lines.append(f"Mean ClauseSpanF1: {stanza_df['ClauseSpanF1'].mean():.4f}")
    report_lines.append("")
    
    # LLM 구문분석 성능 요약
    llm_dep_count = sum(1 for _, row in df.iterrows() if row.get("llm_dependency_results"))
    report_lines.append("LLM Dependency Parsing Performance:")
    report_lines.append("-" * 40)
    report_lines.append(f"Available samples: {llm_dep_count}")
    
    if llm_dep_count > 0:
        llm_dep_data = []
        for _, row in df.iterrows():
            if row.get("llm_dependency_results"):
                for model, results in row["llm_dependency_results"].items():
                    if results.get("metrics"):
                        llm_dep_data.append({
                            "model": model,
                            "UAS": results["metrics"].get("UAS", 0.0),
                            "LAS": results["metrics"].get("LAS", 0.0),
                            "ClauseSpanF1": results["metrics"].get("ClauseSpanF1", 0.0)
                        })
        
        if llm_dep_data:
            llm_dep_df = pd.DataFrame(llm_dep_data)
            for model in llm_dep_df['model'].unique():
                model_data = llm_dep_df[llm_dep_df['model'] == model]
                report_lines.append(f"\n{model}:")
                report_lines.append(f"  Mean UAS: {model_data['UAS'].mean():.4f}")
                report_lines.append(f"  Mean LAS: {model_data['LAS'].mean():.4f}")
                report_lines.append(f"  Mean ClauseSpanF1: {model_data['ClauseSpanF1'].mean():.4f}")
    report_lines.append("")
    
    # LLM 재작성 성능 요약
    llm_rewrite_count = sum(1 for _, row in df.iterrows() if row.get("llm_rewrite_results"))
    report_lines.append("LLM Rewrite Performance:")
    report_lines.append("-" * 40)
    report_lines.append(f"Available samples: {llm_rewrite_count}")
    
    if llm_rewrite_count > 0:
        llm_rewrite_data = []
        for _, row in df.iterrows():
            if row.get("llm_rewrite_results"):
                for model, results in row["llm_rewrite_results"].items():
                    if results.get("best_f1_score", 0.0) > 0:
                        llm_rewrite_data.append({
                            "model": model,
                            "best_f1_score": results.get("best_f1_score", 0.0),
                            "avg_f1_score": results.get("avg_f1_score", 0.0)
                        })
        
        if llm_rewrite_data:
            llm_rewrite_df = pd.DataFrame(llm_rewrite_data)
            for model in llm_rewrite_df['model'].unique():
                model_data = llm_rewrite_df[llm_rewrite_df['model'] == model]
                report_lines.append(f"\n{model}:")
                report_lines.append(f"  Mean Best F1: {model_data['best_f1_score'].mean():.4f}")
                report_lines.append(f"  Mean Avg F1: {model_data['avg_f1_score'].mean():.4f}")
                report_lines.append(f"  Count: {len(model_data)}")
        else:
            report_lines.append("No F1 scores available (LLM rewriting may not have been performed)")
    report_lines.append("")
    
    # 리포트 저장
    with open(output_path / "comprehensive_report.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Comprehensive report saved to {output_path}")

def create_model_specific_visualizations(df: pd.DataFrame, output_dir: str):
    """각 모델별 개별 시각화를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 각 모델별로 개별 시각화 생성
    models_to_analyze = ["gpt-4o", "claude-3-haiku-20240307"]#"gpt-5"
    
    for model in models_to_analyze:
        print(f"Creating visualizations for {model}...")
        
        # 해당 모델의 데이터만 추출
        model_data = []
        
        for _, row in df.iterrows():
            # 구문분석 데이터
            if row.get("llm_dependency_results") and row["llm_dependency_results"].get(model):
                dep_results = row["llm_dependency_results"][model]
                if dep_results.get("metrics"):
                    model_data.append({
                        "id": row["id"],
                        "task": "dependency_parsing",
                        "UAS": dep_results["metrics"].get("UAS", 0.0),
                        "LAS": dep_results["metrics"].get("LAS", 0.0),
                        "ClauseSpanF1": dep_results["metrics"].get("ClauseSpanF1", 0.0)
                    })
            
            # 재작성 데이터
            if row.get("llm_rewrite_results") and row["llm_rewrite_results"].get(model):
                rewrite_results = row["llm_rewrite_results"][model]
                if rewrite_results.get("best_f1_score", 0.0) > 0:
                    model_data.append({
                        "id": row["id"],
                        "task": "text_rewriting",
                        "best_f1_score": rewrite_results.get("best_f1_score", 0.0),
                        "avg_f1_score": rewrite_results.get("avg_f1_score", 0.0)
                    })
        
        if not model_data:
            print(f"No data available for {model}")
            continue
        
        model_df = pd.DataFrame(model_data)
        
        # 모델별 개별 시각화 생성
        create_single_model_visualization(model_df, model, output_path)
        
        # 모델별 상세 통계 저장
        save_model_statistics(model_df, model, output_path)

def create_single_model_visualization(model_df: pd.DataFrame, model_name: str, output_path: Path):
    """단일 모델의 시각화를 생성합니다."""
    # 구문분석 데이터
    dep_data = model_df[model_df['task'] == 'dependency_parsing']
    # 재작성 데이터
    rewrite_data = model_df[model_df['task'] == 'text_rewriting']
    
    # 1. 구문분석 성능 시각화
    if not dep_data.empty:
        plt.figure(figsize=(15, 10))
        
        # UAS 분포
        plt.subplot(2, 3, 1)
        plt.hist(dep_data['UAS'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{model_name} UAS Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('UAS Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # LAS 분포
        plt.subplot(2, 3, 2)
        plt.hist(dep_data['LAS'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title(f'{model_name} LAS Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('LAS Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # ClauseSpanF1 분포
        plt.subplot(2, 3, 3)
        plt.hist(dep_data['ClauseSpanF1'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'{model_name} ClauseSpanF1 Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('ClauseSpanF1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 메트릭 박스플롯
        plt.subplot(2, 3, 4)
        dep_melted = dep_data[['UAS', 'LAS', 'ClauseSpanF1']].melt(var_name='Metric', value_name='Score')
        sns.boxplot(data=dep_melted, x='Metric', y='Score')
        plt.title(f'{model_name} Dependency Metrics', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # UAS vs LAS 스캐터 플롯
        plt.subplot(2, 3, 5)
        plt.scatter(dep_data['UAS'], dep_data['LAS'], alpha=0.6, s=50, color='purple')
        plt.xlabel('UAS Score')
        plt.ylabel('LAS Score')
        plt.title(f'{model_name} UAS vs LAS', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 통계 요약
        plt.subplot(2, 3, 6)
        plt.axis('off')
        stats_text = f"""
        {model_name} Dependency Parsing Statistics:
        
        Total Samples: {len(dep_data)}
        Mean UAS: {dep_data['UAS'].mean():.4f}
        Std UAS: {dep_data['UAS'].std():.4f}
        Min UAS: {dep_data['UAS'].min():.4f}
        Max UAS: {dep_data['UAS'].max():.4f}
        
        Mean LAS: {dep_data['LAS'].mean():.4f}
        Std LAS: {dep_data['LAS'].std():.4f}
        Min LAS: {dep_data['LAS'].min():.4f}
        Max LAS: {dep_data['LAS'].max():.4f}
        
        Mean ClauseSpanF1: {dep_data['ClauseSpanF1'].mean():.4f}
        Std ClauseSpanF1: {dep_data['ClauseSpanF1'].std():.4f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_dependency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 재작성 성능 시각화
    if not rewrite_data.empty:
        plt.figure(figsize=(15, 10))
        
        # F1 점수 분포
        plt.subplot(2, 3, 1)
        plt.hist(rewrite_data['best_f1_score'], bins=20, alpha=0.7, color='coral', edgecolor='black')
        plt.title(f'{model_name} Best F1 Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Best F1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 평균 F1 점수 분포
        plt.subplot(2, 3, 2)
        plt.hist(rewrite_data['avg_f1_score'], bins=20, alpha=0.7, color='gold', edgecolor='black')
        plt.title(f'{model_name} Average F1 Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Average F1 Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Best vs Avg F1 스캐터 플롯
        plt.subplot(2, 3, 3)
        plt.scatter(rewrite_data['avg_f1_score'], rewrite_data['best_f1_score'], alpha=0.6, s=50, color='red')
        plt.xlabel('Average F1 Score')
        plt.ylabel('Best F1 Score')
        plt.title(f'{model_name} Best vs Average F1', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 성능 등급 분포
        plt.subplot(2, 3, 4)
        def classify_f1_score(score):
            if score >= 0.8:
                return 'Excellent (≥0.8)'
            elif score >= 0.6:
                return 'Good (0.6-0.8)'
            elif score >= 0.4:
                return 'Fair (0.4-0.6)'
            else:
                return 'Poor (<0.4)'
        
        rewrite_data['performance_grade'] = rewrite_data['best_f1_score'].apply(classify_f1_score)
        grade_counts = rewrite_data['performance_grade'].value_counts()
        
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
        plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%', colors=colors[:len(grade_counts)])
        plt.title(f'{model_name} Performance Grade Distribution', fontsize=14, fontweight='bold')
        
        # F1 점수 누적 분포
        plt.subplot(2, 3, 5)
        sorted_scores = np.sort(rewrite_data['best_f1_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        plt.plot(sorted_scores, cumulative, linewidth=2, color='red')
        plt.xlabel('Best F1 Score')
        plt.ylabel('Cumulative Probability')
        plt.title(f'{model_name} Cumulative F1 Distribution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 통계 요약
        plt.subplot(2, 3, 6)
        plt.axis('off')
        stats_text = f"""
        {model_name} Rewrite Statistics:
        
        Total Samples: {len(rewrite_data)}
        Mean Best F1: {rewrite_data['best_f1_score'].mean():.4f}
        Std Best F1: {rewrite_data['best_f1_score'].std():.4f}
        Min Best F1: {rewrite_data['best_f1_score'].min():.4f}
        Max Best F1: {rewrite_data['best_f1_score'].max():.4f}
        
        Mean Avg F1: {rewrite_data['avg_f1_score'].mean():.4f}
        Std Avg F1: {rewrite_data['avg_f1_score'].std():.4f}
        
        Performance Grades:
        {grade_counts.to_string()}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_rewrite_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_model_statistics(model_df: pd.DataFrame, model_name: str, output_path: Path):
    """모델별 통계를 저장합니다."""
    # 구문분석 통계
    dep_data = model_df[model_df['task'] == 'dependency_parsing']
    if not dep_data.empty:
        dep_stats = dep_data[['UAS', 'LAS', 'ClauseSpanF1']].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
        dep_stats.to_csv(output_path / f"{model_name}_dependency_statistics.csv")
    
    # 재작성 통계
    rewrite_data = model_df[model_df['task'] == 'text_rewriting']
    if not rewrite_data.empty:
        rewrite_stats = rewrite_data[['best_f1_score', 'avg_f1_score']].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
        rewrite_stats.to_csv(output_path / f"{model_name}_rewrite_statistics.csv")

def create_comprehensive_comparison_table(df: pd.DataFrame, output_dir: str):
    """모든 모델의 결과를 한눈에 볼 수 있는 종합 비교표를 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 모델의 데이터 수집
    all_results = []
    
    # Stanza 데이터
    for _, row in df.iterrows():
        if row.get("stanza_results") and row["stanza_results"].get("metrics"):
            metrics = row["stanza_results"]["metrics"]
            all_results.append({
                "id": row["id"],
                "model": "Stanza",
                "task": "dependency_parsing",
                "UAS": metrics.get("UAS", 0.0),
                "LAS": metrics.get("LAS", 0.0),
                "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0),
                "best_f1_score": None,
                "avg_f1_score": None
            })
    
    # LLM 구문분석 데이터
    for _, row in df.iterrows():
        if row.get("llm_dependency_results"):
            for model, results in row["llm_dependency_results"].items():
                if results.get("metrics"):
                    metrics = results["metrics"]
                    all_results.append({
                        "id": row["id"],
                        "model": model,
                        "task": "dependency_parsing",
                        "UAS": metrics.get("UAS", 0.0),
                        "LAS": metrics.get("LAS", 0.0),
                        "ClauseSpanF1": metrics.get("ClauseSpanF1", 0.0),
                        "best_f1_score": None,
                        "avg_f1_score": None
                    })
    
    # LLM 재작성 데이터
    for _, row in df.iterrows():
        if row.get("llm_rewrite_results"):
            for model, results in row["llm_rewrite_results"].items():
                if results.get("best_f1_score", 0.0) > 0:
                    all_results.append({
                        "id": row["id"],
                        "model": model,
                        "task": "text_rewriting",
                        "UAS": None,
                        "LAS": None,
                        "ClauseSpanF1": None,
                        "best_f1_score": results.get("best_f1_score", 0.0),
                        "avg_f1_score": results.get("avg_f1_score", 0.0)
                    })
    
    if not all_results:
        print("No data available for comprehensive comparison table")
        return
    
    all_results_df = pd.DataFrame(all_results)
    
    # 1. 모델별 성능 요약표 생성
    create_performance_summary_table(all_results_df, output_path)
    
    # 2. 상세 비교표 생성
    create_detailed_comparison_table(all_results_df, output_path)
    
    # 3. 시각화된 비교표 생성
    create_visualized_comparison_table(all_results_df, output_path)

def create_performance_summary_table(all_results_df: pd.DataFrame, output_path: Path):
    """모델별 성능 요약표를 생성합니다."""
    summary_data = []
    
    # 각 모델별로 통계 계산
    for model in all_results_df['model'].unique():
        model_data = all_results_df[all_results_df['model'] == model]
        
        # 구문분석 통계
        dep_data = model_data[model_data['task'] == 'dependency_parsing']
        if not dep_data.empty:
            summary_data.append({
                "Model": model,
                "Task": "Dependency Parsing",
                "Count": len(dep_data),
                "Mean UAS": f"{dep_data['UAS'].mean():.4f}",
                "Std UAS": f"{dep_data['UAS'].std():.4f}",
                "Mean LAS": f"{dep_data['LAS'].mean():.4f}",
                "Std LAS": f"{dep_data['LAS'].std():.4f}",
                "Mean ClauseSpanF1": f"{dep_data['ClauseSpanF1'].mean():.4f}",
                "Std ClauseSpanF1": f"{dep_data['ClauseSpanF1'].std():.4f}",
                "Mean Best F1": "N/A",
                "Std Best F1": "N/A",
                "Mean Avg F1": "N/A",
                "Std Avg F1": "N/A"
            })
        
        # 재작성 통계
        rewrite_data = model_data[model_data['task'] == 'text_rewriting']
        if not rewrite_data.empty:
            summary_data.append({
                "Model": model,
                "Task": "Text Rewriting",
                "Count": len(rewrite_data),
                "Mean UAS": "N/A",
                "Std UAS": "N/A",
                "Mean LAS": "N/A",
                "Std LAS": "N/A",
                "Mean ClauseSpanF1": "N/A",
                "Std ClauseSpanF1": "N/A",
                "Mean Best F1": f"{rewrite_data['best_f1_score'].mean():.4f}",
                "Std Best F1": f"{rewrite_data['best_f1_score'].std():.4f}",
                "Mean Avg F1": f"{rewrite_data['avg_f1_score'].mean():.4f}",
                "Std Avg F1": f"{rewrite_data['avg_f1_score'].std():.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / "performance_summary_table.csv", index=False, encoding='utf-8')
    
    # HTML 형태로도 저장 (더 보기 좋게)
    html_content = f"""
    <html>
    <head>
        <title>Performance Summary Table</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .dependency {{ background-color: #e6f3ff; }}
            .rewrite {{ background-color: #fff2e6; }}
        </style>
    </head>
    <body>
        <h1>Model Performance Summary</h1>
        {summary_df.to_html(classes='dataframe', index=False)}
    </body>
    </html>
    """
    
    with open(output_path / "performance_summary_table.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Performance summary table saved to {output_path}")

def create_detailed_comparison_table(all_results_df: pd.DataFrame, output_path: Path):
    """상세 비교표를 생성합니다."""
    # 각 예제별로 모든 모델의 결과를 비교
    detailed_data = []
    
    for example_id in all_results_df['id'].unique():
        example_data = all_results_df[all_results_df['id'] == example_id]
        
        row_data = {"Example ID": example_id}
        
        # 각 모델별 결과 추가
        for model in example_data['model'].unique():
            model_data = example_data[example_data['model'] == model]
            
            # 구문분석 결과
            dep_data = model_data[model_data['task'] == 'dependency_parsing']
            if not dep_data.empty:
                row_data[f"{model}_UAS"] = f"{dep_data.iloc[0]['UAS']:.4f}"
                row_data[f"{model}_LAS"] = f"{dep_data.iloc[0]['LAS']:.4f}"
                row_data[f"{model}_ClauseSpanF1"] = f"{dep_data.iloc[0]['ClauseSpanF1']:.4f}"
            else:
                row_data[f"{model}_UAS"] = "N/A"
                row_data[f"{model}_LAS"] = "N/A"
                row_data[f"{model}_ClauseSpanF1"] = "N/A"
            
            # 재작성 결과
            rewrite_data = model_data[model_data['task'] == 'text_rewriting']
            if not rewrite_data.empty:
                row_data[f"{model}_BestF1"] = f"{rewrite_data.iloc[0]['best_f1_score']:.4f}"
                row_data[f"{model}_AvgF1"] = f"{rewrite_data.iloc[0]['avg_f1_score']:.4f}"
            else:
                row_data[f"{model}_BestF1"] = "N/A"
                row_data[f"{model}_AvgF1"] = "N/A"
        
        detailed_data.append(row_data)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_path / "detailed_comparison_table.csv", index=False, encoding='utf-8')
    
    print(f"Detailed comparison table saved to {output_path}")

def create_visualized_comparison_table(all_results_df: pd.DataFrame, output_path: Path):
    """시각화된 비교표를 생성합니다."""
    # 1. 구문분석 성능 비교 히트맵
    dep_data = all_results_df[all_results_df['task'] == 'dependency_parsing']
    if not dep_data.empty:
        plt.figure(figsize=(12, 8))
        
        # 모델별 평균 성능 계산
        dep_summary = dep_data.groupby('model')[['UAS', 'LAS', 'ClauseSpanF1']].mean()
        
        # 히트맵 생성
        sns.heatmap(dep_summary, annot=True, cmap='YlOrRd', fmt='.4f', cbar_kws={'label': 'Score'})
        plt.title('Dependency Parsing Performance Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Model')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.savefig(output_path / "dependency_parsing_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 재작성 성능 비교 히트맵
    rewrite_data = all_results_df[all_results_df['task'] == 'text_rewriting']
    if not rewrite_data.empty:
        plt.figure(figsize=(10, 6))
        
        # 모델별 평균 성능 계산
        rewrite_summary = rewrite_data.groupby('model')[['best_f1_score', 'avg_f1_score']].mean()
        
        # 히트맵 생성
        sns.heatmap(rewrite_summary, annot=True, cmap='YlGnBu', fmt='.4f', cbar_kws={'label': 'F1 Score'})
        plt.title('Text Rewriting Performance Heatmap', fontsize=16, fontweight='bold')
        plt.ylabel('Model')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.savefig(output_path / "text_rewriting_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. 종합 성능 비교 차트
    plt.figure(figsize=(15, 10))
    
    # 구문분석 성능 비교
    plt.subplot(2, 2, 1)
    if not dep_data.empty:
        dep_summary = dep_data.groupby('model')[['UAS', 'LAS', 'ClauseSpanF1']].mean()
        dep_summary.plot(kind='bar', ax=plt.gca())
        plt.title('Dependency Parsing Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 재작성 성능 비교
    plt.subplot(2, 2, 2)
    if not rewrite_data.empty:
        rewrite_summary = rewrite_data.groupby('model')[['best_f1_score', 'avg_f1_score']].mean()
        rewrite_summary.plot(kind='bar', ax=plt.gca())
        plt.title('Text Rewriting Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 모델별 샘플 수 비교
    plt.subplot(2, 2, 3)
    sample_counts = all_results_df.groupby(['model', 'task']).size().unstack(fill_value=0)
    sample_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Sample Count by Model and Task', fontsize=14, fontweight='bold')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 성능 등급 분포 비교
    plt.subplot(2, 2, 4)
    if not rewrite_data.empty:
        def classify_f1_score(score):
            if score >= 0.8:
                return 'Excellent (≥0.8)'
            elif score >= 0.6:
                return 'Good (0.6-0.8)'
            elif score >= 0.4:
                return 'Fair (0.4-0.6)'
            else:
                return 'Poor (<0.4)'
        
        rewrite_data['performance_grade'] = rewrite_data['best_f1_score'].apply(classify_f1_score)
        grade_by_model = pd.crosstab(rewrite_data['model'], rewrite_data['performance_grade'])
        
        colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347']
        grade_by_model.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors[:len(grade_by_model.columns)])
        plt.title('Performance Grade Distribution by Model', fontsize=14, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Performance Grade')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "comprehensive_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualized comparison tables saved to {output_path}")

def main():
    """메인 함수"""
    # 입력 파일 경로
    enhanced_results_file = "clause_dep/results/detailed_results.jsonl"
    output_dir = "clause_dep/results/stat_result"
    
    # 파일 존재 확인
    if not Path(enhanced_results_file).exists():
        print(f"Error: {enhanced_results_file} file not found.")
        print("Please run enhanced_parser_evaluation.py first.")
        return
    
    print(f"Loading enhanced results: {enhanced_results_file}")
    df = load_enhanced_results(enhanced_results_file)
    
    print(f"Loaded {len(df)} samples.")
    
    # 시각화 생성
    print("Creating Stanza parser visualizations...")
    create_stanza_parser_visualizations(df, output_dir)
    
    print("Creating LLM dependency parsing visualizations...")
    create_llm_dependency_visualizations(df, output_dir)
    
    print("Creating LLM rewrite visualizations...")
    create_llm_rewrite_visualizations(df, output_dir)
    
    print("Creating comprehensive comparison visualizations...")
    create_comprehensive_comparison(df, output_dir)
    
    print("Creating metrics distribution visualizations...")
    create_metrics_distribution_visualizations(df, output_dir)
    
    print("Creating metrics boxplot visualizations...")
    create_metrics_boxplot_visualizations(df, output_dir)
    
    print("Creating correlation heatmap...")
    create_correlation_heatmap(df, output_dir)
    
    print("Creating comprehensive report...")
    create_comprehensive_report(df, output_dir)
    
    print("Creating model-specific visualizations...")
    create_model_specific_visualizations(df, output_dir)

    print("Creating comprehensive comparison table...")
    create_comprehensive_comparison_table(df, output_dir)
    
    print("All visualizations completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
