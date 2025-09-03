
# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple, Set

def parse_conllu_lines(conllu_text: str) -> List[Dict]:
    """CoNLL-U 형식의 텍스트를 파싱합니다."""
    rows = []
    lines = conllu_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # CoNLL-U 형식: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
        parts = line.split('\t')
        if len(parts) >= 8:
            try:
                row = {
                    "id": int(parts[0]),
                    "form": parts[1],
                    "lemma": parts[2],
                    "upos": parts[3],
                    "head": int(parts[6]),  # HEAD는 7번째 컬럼 (인덱스 6)
                    "deprel": parts[7]      # DEPREL은 8번째 컬럼 (인덱스 7)
                }
                rows.append(row)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue
    
    return rows

def uas_las(pred: List[Dict], gold: List[Dict]) -> Tuple[float, float]:
    # Match by token index (ID). If lengths differ, compare up to min length.
    n = min(len(pred), len(gold))
    if n == 0:
        return 0.0, 0.0
    uas_cnt = sum(1 for i in range(n) if pred[i]["head"] == gold[i]["head"])
    las_cnt = sum(1 for i in range(n) if pred[i]["head"] == gold[i]["head"] and pred[i]["deprel"] == gold[i]["deprel"])
    return uas_cnt / n, las_cnt / n

def build_children(rows: List[Dict]) -> Dict[int, List[int]]:
    ch = {r["id"]: [] for r in rows}
    for r in rows:
        if r["head"] in ch:
            ch[r["head"]].append(r["id"])
    return ch

def subtree_ids(root: int, children: Dict[int, List[int]]) -> Set[int]:
    out = set([root]); stack = [root]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, []):
            if c not in out:
                out.add(c); stack.append(c)
    return out

def clause_span_f1(pred: List[Dict], gold: List[Dict], target_labels=("advcl","acl:relcl","ccomp","csubj")) -> float:
    # For each clause head in gold, compute span F1 against best matching pred span of same label; average.
    if not pred or not gold:
        return 0.0
    import itertools
    g_children = build_children(gold)
    p_children = build_children(pred)
    g_clauses = [(r["id"], r["deprel"], subtree_ids(r["id"], g_children)) for r in gold if r["deprel"] in target_labels or r["deprel"].startswith("acl:relcl")]
    p_clauses = [(r["id"], r["deprel"], subtree_ids(r["id"], p_children)) for r in pred if r["deprel"] in target_labels or r["deprel"].startswith("acl:relcl")]
    if not g_clauses:
        return 1.0  # nothing to match
    def f1(a: Set[int], b: Set[int]) -> float:
        inter = len(a & b)
        if inter == 0:
            return 0.0
        prec = inter / len(b) if b else 0.0
        rec = inter / len(a) if a else 0.0
        return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    scores = []
    for gid, glabel, gspan in g_clauses:
        # best same-label; if none, best overall
        candidates = [c for c in p_clauses if c[1] == glabel]
        if not candidates:
            candidates = p_clauses
        best = 0.0
        for pid, plabel, pspan in candidates:
            best = max(best, f1(gspan, pspan))
        scores.append(best)
    return sum(scores)/len(scores) if scores else 0.0
