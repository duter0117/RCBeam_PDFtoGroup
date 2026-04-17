import re

def normalize_text(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    if s in ("LLM沒有東西", "LLM看不出來"):
        return ""
    s = s.strip()
    s = s.replace("（", "(").replace("）", ")").replace("×", "x").replace("Ｘ", "x").replace("X", "x")
    s = re.sub(r'\s+', ' ', s)
    
    # ==== 自定義同義詞轉換與正規化 ====
    # 1. 過濾掉附帶尺寸的括號雜訊 (例如 B3F B4-3a (50x70) -> B3F B4-3a)
    # 判斷：如果這個字串本身不是「純尺寸」，才把尺寸附屬物刪除。如果是純尺寸，只去掉括號
    is_pure_dimension = re.match(r'^\s*\(?\s*\d+\s*[xX*]\s*\d+\s*\)?\s*$', s, flags=re.IGNORECASE)
    if not is_pure_dimension:
        s = re.sub(r'\s*\(\s*\d+\s*[xX*]\s*\d+\s*\)', '', s, flags=re.IGNORECASE).strip()
    else:
        s = s.replace("(", "").replace(")", "").strip()
    
    # (原先處理 1-#4@20 的規則已廢棄，遵循工程圖面上「無標示根數即為不明/預設」的本意)
    
    # 2. OCR 吃空白修復：樓層代號(B4F/RF)與梁代號(FB3/G1)黏在一起時，自動補回空格
    # B4FFB3-2 → B4F FB3-2, B3FG1-2 → B3F G1-2, RFCB1 → RF CB1
    s = re.sub(r'((?:R|B?\d+)[Ff])([A-Za-z])', r'\1 \2', s)
    
    # 3. 處理腰筋標註標準化 (確保 E.F. 前有空白並統一格式)
    s = re.sub(r'\(\s*[Ee]\.?[Ff]\.?\s*\)', '(E.F.)', s)
    s = re.sub(r'([^\s])\(E\.F\.\)', r'\1 (E.F.)', s)
    s = re.sub(r'_\s*\(E\.F\.\)', ' (E.F.)', s)  # 修復 OCR 誤判底線
    
    # 4. 修復鋼筋數量與號數之間漏掉的減號 (例如 6#11 → 6-#11)
    s = re.sub(r'(?<![-])(\d+)\s*(#\d+)', r'\1-\2', s)
    
    # 5. 去除程式內部產生的重名標籤 (如: (重複-2))
    s = re.sub(r'\s*\(重複-\d+\)\s*$', '', s)
    
    # 5. 直接替換同義詞的字典
    synonyms = {
        "e.f": "E.F.",
        "ef": "E.F.",
        "E.F": "E.F.",
        "EF": "E.F.",
    }
    
    if s in synonyms:
        s = synonyms[s]
        
    return s

def normalize_list(lst) -> list:
    if not lst or not isinstance(lst, list):
        return []
    result = [normalize_text(str(x)) for x in lst if x]
    # 去除空字串
    result = [x for x in result if x]
    # 去重 (保留順序)
    seen = set()
    deduped = []
    for x in result:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    # 鋼筋排序：數量多的排前面 (如 14-#11 排在 3-#11 前面)
    def rebar_sort_key(s):
        m = re.match(r'^(\d+)-#', s)
        return -int(m.group(1)) if m else 0
    deduped.sort(key=rebar_sort_key)
    return deduped

def normalize_dict(d: dict) -> dict:
    if not d or not isinstance(d, dict):
        return d
    
    normalized = {}
    for k, v in d.items():
        if isinstance(v, str):
            # Certain fields should not be heavily normalized (like IDs, but uppercase is fine for now)
            # Actually our format requires everything to be uppercase/normalized.
            normalized[k] = normalize_text(v)
        elif isinstance(v, list):
            # Might be List[str] or nested
            normalized[k] = [normalize_text(x) if isinstance(x, str) else normalize_dict(x) if isinstance(x, dict) else x for x in v]
        elif isinstance(v, dict):
            normalized[k] = normalize_dict(v)
        else:
            normalized[k] = v
            
    return normalized
