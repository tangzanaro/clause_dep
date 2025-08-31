
# -*- coding: utf-8 -*-
from typing import Dict, Any, List
from prompts import STRUCTURE_SYSTEM_PROMPT, structure_user_prompt, REWRITE_SYSTEM_PROMPT, rewrite_user_prompt
from dep_utils import parse_conllu_lines, uas_las, clause_span_f1

def run_parsing_example(client, sentence: str, **gen_kwargs) -> Dict[str, Any]:
    sys_p = STRUCTURE_SYSTEM_PROMPT
    usr_p = structure_user_prompt(sentence)
    raw = client.run(sys_p, usr_p, mode="parse", **gen_kwargs)
    pred = parse_conllu_lines(raw)
    return {"raw": raw, "pred": pred}

def run_rewrite_example(client, sentence: str, instruction: str, **gen_kwargs) -> Dict[str, Any]:
    sys_p = REWRITE_SYSTEM_PROMPT
    usr_p = rewrite_user_prompt(sentence, instruction)
    out = client.run(sys_p, usr_p, mode="rewrite", **gen_kwargs)
    return {"raw": out, "pred_text": out}

def eval_parsing(pred_rows, gold_rows) -> Dict[str, float]:
    uas, las = uas_las(pred_rows, gold_rows)
    span = clause_span_f1(pred_rows, gold_rows)
    return {"UAS": uas, "LAS": las, "ClauseSpanF1": span}

def simple_text_metrics(hyp: str, ref: str) -> Dict[str, float]:
    import math
    htoks = hyp.strip().split()
    rtoks = ref.strip().split()
    # token F1
    inter = len(set(htoks) & set(rtoks))
    prec = inter/len(set(htoks)) if htoks else 0.0
    rec = inter/len(set(rtoks)) if rtoks else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    # Levenshtein distance normalized
    import numpy as np
    def edit(a, b):
        dp = [[i+j if i*j==0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
        for i in range(1,len(a)+1):
            dp[i][0] = i
        for j in range(1,len(b)+1):
            dp[0][j] = j
        for i in range(1,len(a)+1):
            for j in range(1,len(b)+1):
                cost = 0 if a[i-1]==b[j-1] else 1
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
        return dp[-1][-1]
    ed = edit(htoks, rtoks)
    norm_ed = 1.0 - (ed / max(1, len(htoks)+len(rtoks)))
    return {"TokenF1": f1, "NormEdit": norm_ed}

def label_errors(pred_rows, gold_rows) -> List[str]:
    labels = []
    # boundary: span F1 below threshold AND clause exists in gold
    span = clause_span_f1(pred_rows, gold_rows)
    g_has_clause = any(r["deprel"] in ("advcl","ccomp","csubj") or r["deprel"].startswith("acl:relcl") for r in gold_rows)
    if g_has_clause and span < 0.6:
        labels.append("BOUNDARY")
    # head/label confusion: LAS-UAS gap or low UAS
    from statistics import mean
    uas, las = uas_las(pred_rows, gold_rows)
    if uas < 0.8:
        labels.append("HEAD")
    if abs(uas - las) > 0.1:
        labels.append("LABEL")
    # missing/extra markers (heuristic: root count or presence of advcl/ccomp/acl mismatch)
    g_clause = sum(1 for r in gold_rows if r["deprel"] in ("advcl","ccomp","csubj") or r["deprel"].startswith("acl:relcl"))
    p_clause = sum(1 for r in pred_rows if r["deprel"] in ("advcl","ccomp","csubj") or r["deprel"].startswith("acl:relcl"))
    if p_clause < g_clause:
        labels.append("MISSING")
    elif p_clause > g_clause:
        labels.append("EXTRA")
    return sorted(set(labels))
