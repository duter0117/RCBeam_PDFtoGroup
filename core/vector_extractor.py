import os
import io
import json
import fitz
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont
from core.debug_logger import debug_print

print = debug_print


class VectorExtractor:
    def __init__(self, pdf_bytes: bytes):
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.doc:
            self.doc.close()
            self.doc = None

    def __del__(self):
        if self.doc:
            self.doc.close()

    def extract_page_data(self, page_num: int = 0) -> Dict[str, Any]:
        """
        提取 PDF 指定頁面的所有向量線段與純文字塊
        避開圖片失真，直接抓取 CAD 轉出時的底層幾何座標。
        """
        if page_num >= len(self.doc):
            return {"error": "Page number out of range"}
            
        page = self.doc[page_num]
        
        # 1. 提取所有向量幾何 (線段、矩形 等 CAD Path)
        drawings = page.get_drawings()
        vectors = []
        for d in drawings:
            # 簡化記錄每個繪圖物件的邊界框與屬性
            vectors.append({
                "type": "path",
                "rect": [round(x, 2) for x in d["rect"]], # [x0, y0, x1, y1]
                "color": d.get("color"),
                "width": d.get("width")
            })
            
        # 2. 提取所有文字塊
        text_blocks = page.get_text("blocks")
        texts = []
        for b in text_blocks:
            # b format: (x0, y0, x1, y1, "text", block_no, block_type)
            if b[6] == 0: # type 0 is text block
                text_content = b[4].strip()
                if text_content:
                    texts.append({
                        "rect": [round(x, 2) for x in b[:4]],
                        "text": text_content
                    })
                
        return {
            "page_num": page_num,
            "width": round(page.rect.width, 2),
            "height": round(page.rect.height, 2),
            "vector_count": len(vectors),
            "text_count": len(texts),
            "texts_data": texts,
            "vectors_sample": vectors[:50]
        }

    def find_beam_bboxes_heuristic(self, page_num: int = 0) -> list:
        """
        Phase 3 核心演算法：自適應幾何尋邊 (Micro-Vision Crop Box)
        利用文字 Regex 定位梁標題，並往上搜索幾何邊界，使用聯集算出精確 Bounding Box。
        """
        import re
        page = self.doc[page_num]
        blocks = page.get_text("blocks")
        
        # 1. 抓出所有潛在的梁標題 (特徵：結尾伴隨著尺寸，如 B1F FWB1 (100x500))
        titles = []
        for b in blocks:
            if b[6] == 0:
                txt = b[4].strip()
                # 過濾含有 "(數字x數字)" 的工程標題常態字串
                if re.search(r'[a-zA-Z0-9_-]+', txt) and re.search(r'\([0-9]+[xX*][0-9]+\)', txt):
                    titles.append({"text": txt, "rect": b[:4]})
                    
        # 2. 為每個標題動態尋找涵蓋其配筋繪圖的 Bounding Box
        drawings = page.get_drawings()
        vectors = [d["rect"] for d in drawings]
        
        results = []
        for t in titles:
            tx0, ty0, tx1, ty1 = t["rect"]
            title_width = tx1 - tx0
            
            # 定義搜尋範圍閾值 (防呆機制：最多往上找 6 倍寬度，左右寬容 2.5 倍寬度)
            max_search_height = max(title_width * 6.0, 300) # 給個合理極限下限
            search_area = fitz.Rect(
                tx0 - (title_width * 2.5), 
                ty0 - max_search_height, 
                tx1 + (title_width * 2.5), 
                ty1 + 20 # 稍微往下包一點避免切到下方文字
            )
            
            # 蒐集落在此預估勢力範圍內的所有向量幾何
            contained_rects = []
            for v_rect in vectors:
                vr = fitz.Rect(v_rect)
                if vr.intersects(search_area):
                    contained_rects.append(vr)
                    
            # 幾何連通集計算：計算這些落網線條的聯集 (Rectangle Union)
            if contained_rects:
                final_box = contained_rects[0]
                for vr in contained_rects[1:]:
                    final_box = final_box | vr
                
                # 防呆閥值機制：萬一連通集爆掉 (幾何範圍失控交疊到隔壁)，強制裁斷其越境長度
                if final_box.height > max_search_height * 1.5:
                    final_box.y0 = ty0 - max_search_height
                    
                # 確保標題本身一定在框內
                final_box = final_box | fitz.Rect(tx0, ty0, tx1, ty1)
            else:
                # 萬一這隻梁只有文字沒有畫圖? 就退回安全搜尋框
                final_box = search_area
                
            # 將最終完美的畫框擴大 15 pixel 作為邊界緩衝
            final_box.x0 -= 15
            final_box.y0 -= 15
            final_box.x1 += 15
            final_box.y1 += 15
            
            results.append({
                "beam_id": t["text"],
                "anchor_rect": t["rect"],
                "adaptive_bbox": [round(final_box.x0, 2), round(final_box.y0, 2), round(final_box.x1, 2), round(final_box.y1, 2)]
            })
            
        return results

    @staticmethod
    def _content_trim_bboxes(bboxes, thresh, page_w, page_h,
                             pad_x=60, pad_y=20, trim_bottom=False):
        """
        Content-Aware Trim: 使用二值化圖 (thresh, 4x scale) 收緊 bbox 邊界。
        
        Args:
            bboxes: list of [x0, y0, x1, y1] in PDF units (mutated in-place)
            thresh: 二值圖 (4x scale, 白=有墨水)
            page_w, page_h: 頁面尺寸 (PDF units), 用於 clamp
            pad_x: X 軸緩衝 (4x px), 預設 60 ≈ 15pt
            pad_y: Y 軸緩衝 (4x px), 預設 20 ≈ 5pt
            trim_bottom: 是否也收緊底部 (母塊階段 False, Pass2 階段 True)
        """
        import numpy as np
        for bbox in bboxes:
            # (1) Page clamp
            bbox[0] = max(0.0, bbox[0])
            bbox[1] = max(0.0, bbox[1])
            bbox[2] = min(page_w, bbox[2])
            bbox[3] = min(page_h, bbox[3])
            
            # (2) Content trim via binary projection
            px0 = max(0, int(bbox[0] * 4))
            py0 = max(0, int(bbox[1] * 4))
            px1 = min(thresh.shape[1], int(bbox[2] * 4))
            py1 = min(thresh.shape[0], int(bbox[3] * 4))
            if px1 <= px0 or py1 <= py0:
                continue
            
            roi = thresh[py0:py1, px0:px1]
            if roi.size == 0:
                continue
            
            # X 軸
            col_sums = roi.sum(axis=0)
            nonzero_cols = np.where(col_sums > 0)[0]
            if len(nonzero_cols) > 0:
                trim_left  = max(0,              nonzero_cols[0]  - pad_x)
                trim_right = min(px1 - px0 - 1, nonzero_cols[-1] + pad_x)
                bbox[0] = (px0 + trim_left)  / 4.0
                bbox[2] = (px0 + trim_right) / 4.0
            
            # Y 軸頂部
            row_sums = roi.sum(axis=1)
            nonzero_rows = np.where(row_sums > 0)[0]
            if len(nonzero_rows) > 0:
                trim_top = max(0, nonzero_rows[0] - pad_y)
                bbox[1] = (py0 + trim_top) / 4.0
                # Y 軸底部 (僅在 trim_bottom=True 時)
                if trim_bottom:
                    trim_bot = min(py1 - py0 - 1, nonzero_rows[-1] + pad_y)
                    bbox[3] = (py0 + trim_bot) / 4.0

    def _nms_bboxes(self, bboxes: list, iou_thresh: float = 0.5) -> tuple[list, list]:
        """
        Non-Maximum Suppression：過濾高度重疊的 bbox，避免同一張圖被傳給 Gemini 多次。
        輸入按面積由大到小排序，優先保留較大的框。
        """
        if not bboxes:
            return [], []
        # 按面積降序，保留大框優先
        sorted_boxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        keep = []
        drops = []
        for bbox in sorted_boxes:
            suppressed = False
            for k in keep:
                inter_x0 = max(bbox[0], k[0])
                inter_y0 = max(bbox[1], k[1])
                inter_x1 = min(bbox[2], k[2])
                inter_y1 = min(bbox[3], k[3])
                inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
                area_small = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                # 這邊改用 IoA (Intersection over Area_min)，因為 bbox 是依面積降序排列的，
                # 所以 bbox 是現在要檢查的比較小的框。只要小框有大部分被大框(k)吃掉，就殺掉！
                ioa = inter / (area_small + 1e-6)
                if ioa > iou_thresh:
                    suppressed = True
                    # 融合 (Merge): 只要超過閾值，就把小框(bbox)的面積撐進大框(k)裡
                    k[0] = min(k[0], bbox[0])
                    k[1] = min(k[1], bbox[1])
                    k[2] = max(k[2], bbox[2])
                    k[3] = max(k[3], bbox[3])
                    break
            if not suppressed:
                keep.append(bbox)
            else:
                drops.append(bbox)
        return keep, drops

    def extract_opencv_bboxes(self, page_num: int = 0, cv_params: dict = None) -> tuple[list, dict]:
        """
        Phase 3: OpenCV 形態學尋邊 (Morphological Bounding)
        接收動態 cv_params (dilation_iterations, min_area, padding_bottom) 取代寫死的魔法數字。
        最後執行 NMS 去除重疊的父子輪廓，避免重複送圖給 Gemini。
        """
        if cv_params is None:
            cv_params = {}
            
        dilation_iterations = int(cv_params.get('dilation_iterations', 2))
        min_area = int(cv_params.get('min_area', 100000))
        padding_bottom = int(cv_params.get('padding_bottom', 160))
        import cv2
        import numpy as np
        
        page = self.doc[page_num]
        
        # 放大渲染做二值化處理
        mat = fitz.Matrix(4.0, 4.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        
        # 黑白轉換與膨脹
        _, thresh = cv2.threshold(img_data, 150, 255, cv2.THRESH_BINARY_INV)
        
        # --- Hough Transform 邊框清除器 ---
        hough_threshold_pct = int(cv_params.get('hough_threshold', 95)) / 100.0
        # 允許的斷線間隙放大一點，因為有時候圖框線會被跨過的字截斷
        gap_limit = 100 
        
        h_len = int(pix.width * hough_threshold_pct)
        v_len = int(pix.height * hough_threshold_pct)
        
        # 找尋影像中所有的極長直線
        min_search_length = min(h_len, v_len)
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, threshold=min_search_length//2, minLineLength=min_search_length, maxLineGap=gap_limit)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                
                # 嚴格判斷方向與佔比：
                # 1. 如果是水平線 (X跨度極大，Y沒什麼變)，就用寬度 (h_len) 來當標準
                is_horizontal = (dx >= h_len) and (dy < 50)
                # 2. 如果是垂直線 (Y跨度極大，X沒什麼變)，就用高度 (v_len) 來當標準
                is_vertical = (dy >= v_len) and (dx < 50)
                
                if is_horizontal or is_vertical:
                    cv2.line(thresh, (x1, y1), (x2, y2), 0, thickness=20)
        # -----------------------------------
        
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=dilation_iterations)
        
        # 儲存膨脹後的海島圖供使用者視覺除錯
        os.makedirs("crops", exist_ok=True)
        img_island = Image.fromarray(dilated)
        img_island.save(f"crops/debug_islands_page_{page_num}.png")
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        total_contours = len(contours)
        noise_dropped = 0
        pre_nms_results = []
        
        # 記錄要存檔除錯的跌落圖塊
        dropped_for_save = []
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # 過濾掉雜訊、單獨文字體，以及超級大的頁框
            if area > min_area and area < (pix.width * pix.height * 0.25):
                # 退回給 PDF 的原始單位坐標 (除以 4.0)
                orig_x0 = max(0, (x - 40) / 4.0)
                orig_y0 = max(0, (y - 40) / 4.0)
                orig_x1 = min(page.rect.width, (x + w + 40) / 4.0)
                orig_y1 = min(page.rect.height, (y + h + padding_bottom) / 4.0)
                pre_nms_results.append([orig_x0, orig_y0, orig_x1, orig_y1])
            else:
                noise_dropped += 1
                # 若面積過大 (> 25%)，很可能是誤判為整張大圖框而被丟棄者，必須存檔供檢閱
                if area >= (pix.width * pix.height * 0.25):
                    orig_x0 = max(0, (x - 10) / 4.0)
                    orig_y0 = max(0, (y - 10) / 4.0)
                    orig_x1 = min(page.rect.width, (x + w + 10) / 4.0)
                    orig_y1 = min(page.rect.height, (y + h + 10) / 4.0)
                    dropped_for_save.append(("oversize", [orig_x0, orig_y0, orig_x1, orig_y1]))
                # 對於面積實在太小(例如只有單獨文字、碎點，面積 < 4000)者直接放生不存檔，否則會跑出幾千張圖拖垮系統
                elif area > 4000:
                    orig_x0 = max(0, (x - 10) / 4.0)
                    orig_y0 = max(0, (y - 10) / 4.0)
                    orig_x1 = min(page.rect.width, (x + w + 10) / 4.0)
                    orig_y1 = min(page.rect.height, (y + h + 10) / 4.0)
                    dropped_for_save.append(("noise", [orig_x0, orig_y0, orig_x1, orig_y1]))
        
        pre_nms_len = len(pre_nms_results)
        
        # NMS 去重後，再按 y, x 排序符合人類閱讀習慣
        results, nms_drops = self._nms_bboxes(pre_nms_results, iou_thresh=0.4)
        post_nms_len = len(results)
        nms_dropped = pre_nms_len - post_nms_len
        
        # 儲存 NMS 被刷掉的圖塊 (可能非常有價值，因為是被判斷為重複區域而刷掉)
        for nd in nms_drops:
            dropped_for_save.append(("nms", nd))
            
        results.sort(key=lambda b: (b[1], b[0]))
        
        # === Phase 3.4: X 軸投影斷裂分割 (Column Projection Split) ===
        # 在搜尋標題之前，用二值圖的垂直投影檢查每個母塊是否在 X 軸上有完全斷開的區域。
        # 如果有，將它拆分成左右各自獨立的子塊。
        split_results = []
        phase34_split_count = 0
        min_gap_px = 80  # 4x scale: 80px = 20pt，至少要有 20pt 的空白才算斷裂
        
        for bbox in results:
            x0, y0, x1, y1 = bbox
            # 轉回 4x pixel 座標
            px0 = max(0, int(x0 * 4.0))
            py0 = max(0, int(y0 * 4.0))
            px1 = min(thresh.shape[1], int(x1 * 4.0))
            py1 = min(thresh.shape[0], int(y1 * 4.0))
            
            if px1 - px0 < 100 or py1 - py0 < 20:
                split_results.append(bbox)
                continue
            
            roi = thresh[py0:py1, px0:px1]
            # 垂直投影：每一欄的白色像素加總
            col_proj = np.sum(roi > 0, axis=0)
            
            # 找出所有空白欄 (投影值 = 0) 的連續區段
            is_empty = (col_proj == 0).astype(np.int8)
            gaps = []
            gap_start = None
            for ci in range(len(is_empty)):
                if is_empty[ci] == 1 and gap_start is None:
                    gap_start = ci
                elif is_empty[ci] == 0 and gap_start is not None:
                    if ci - gap_start >= min_gap_px:
                        gaps.append((gap_start, ci))
                    gap_start = None
            if gap_start is not None and len(is_empty) - gap_start >= min_gap_px:
                gaps.append((gap_start, len(is_empty)))
            
            if not gaps:
                split_results.append(bbox)
                continue
            
            # 用找到的斷裂帶切成多個子塊
            phase34_split_count += 1
            prev_x = 0
            sub_bboxes = []
            for gap_s, gap_e in gaps:
                if gap_s > prev_x:
                    sub_x0 = x0 + prev_x / 4.0
                    sub_x1 = x0 + gap_s / 4.0
                    sub_bboxes.append([sub_x0, y0, sub_x1, y1])
                prev_x = gap_e
            # 最後一段
            if prev_x < (px1 - px0):
                sub_x0 = x0 + prev_x / 4.0
                sub_bboxes.append([sub_x0, y0, x1, y1])
            
            # 過濾掉太窄的碎片 (< 30pt)
            for sb in sub_bboxes:
                if sb[2] - sb[0] > 30:
                    split_results.append(sb)
        
        if phase34_split_count > 0:
            print(f"[Phase 3.4] X 軸投影斷裂分割: 將 {len(results)} 個母塊拆分為 {len(split_results)} 個")
        results = split_results
        results.sort(key=lambda b: (b[1], b[0]))

                # === Phase 3.5.5: Global Title Collection & LLM Filter ===
        all_potential_titles = []
        global_title_id = 0
        confirmed_titles_list = []
        
        import re
        def is_title_candidate(text):
            text = text.strip()
            if len(text) < 2: return False
            if '@' in text or '#' in text: return False
            if re.search(r'\d+\s*[xX×*]\s*\d+', text): return True
            if re.search(r'[A-Za-z]', text): return True
            return False

        enable_decomp = cv_params.get('enable_decomp', True)
        if enable_decomp:
            os.makedirs("crops/rough_cut_pass1", exist_ok=True)
            with open("crops/rough_cut_pass1/titles_log.txt", "w", encoding="utf-8") as _f:
                _f.write("=== 初切標題記錄 ===\n\n")
                
            for p_idx, bbox in enumerate(results):
                orig_x0, orig_y0, orig_x1, orig_y1 = bbox
                
                max_search_y = min(orig_y1 + 150, page.rect.height)
                # 排除重疊
                for j, other_bbox in enumerate(results):
                    if j != p_idx and other_bbox[1] > orig_y1:
                        o_x0, o_x1 = other_bbox[0], other_bbox[2]
                        overlap_x = max(0, min(orig_x1, o_x1) - max(orig_x0, o_x0))
                        if overlap_x > (orig_x1 - orig_x0) * 0.1:
                            max_search_y = min(max_search_y, other_bbox[1] + 5)
                            
                search_y0 = orig_y0 - 20
                search_y1 = max(orig_y1 + 10, max_search_y)
                search_rect = fitz.Rect(orig_x0 - 60, search_y0, orig_x1 + 60, search_y1)
                
                potential_titles = []
                raw_titles = []
                if search_rect.width > 10 and search_rect.height > 10:
                    try:
                        if not hasattr(self, '_title_ocr'):
                            from rapidocr_openvino import RapidOCR
                            self._title_ocr = RapidOCR()
                        
                        search_pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=search_rect)
                        channels = search_pix.n
                        search_img = np.frombuffer(search_pix.samples, dtype=np.uint8).reshape(
                            search_pix.height, search_pix.width, channels
                        )
                        if channels == 1:
                            search_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
                        elif channels == 4:
                            search_img = cv2.cvtColor(search_img, cv2.COLOR_BGRA2RGB)
                            
                        import numpy as np
                        import cv2
                        ocr_result, _ = self._title_ocr(search_img)
                        
                        if ocr_result:
                            for idx, (ocr_bbox, ocr_text, ocr_conf) in enumerate(ocr_result):
                                if ocr_conf > 0.5 and is_title_candidate(ocr_text):
                                    ys = [pt[1] for pt in ocr_bbox]
                                    xs = [pt[0] for pt in ocr_bbox]
                                    
                                    # 轉換為 Full-Page Pixel coords (4x scale)
                                    abs_cx = sum(xs)/len(xs) + search_rect.x0 * 4.0
                                    abs_cy = sum(ys)/len(ys) + search_rect.y0 * 4.0
                                    abs_bottom = max(ys) + search_rect.y0 * 4.0
                                    
                                    raw_titles.append({
                                        "text": ocr_text,
                                        "cx": abs_cx,
                                        "cy": abs_cy,
                                        "bottom_y": abs_bottom,
                                        "h": max(ys) - min(ys),
                                        "w": max(xs) - min(xs),
                                        "ocr_x_left": min(xs) / 4.0 + search_rect.x0,
                                        "ocr_x_right": max(xs) / 4.0 + search_rect.x0
                                    })
                                    
                        # Spatial Merging
                        for rt in raw_titles:
                            merged = False
                            for pt in potential_titles:
                                horizontal_gap = max(0, max(rt["ocr_x_left"], pt["ocr_x_left"]) - min(rt["ocr_x_right"], pt["ocr_x_right"]))
                                if horizontal_gap < 60 and abs(rt["cy"] - pt["cy"]) < 30:
                                    # 合併前檢查：如果合併後會出現 2 組以上尺寸標記，代表是獨立標題，不合併
                                    combined_text = pt["text"] + " " + rt["text"]
                                    dim_count = len(re.findall(r'\d+\s*[xX×*]\s*\d+', combined_text))
                                    if dim_count >= 2:
                                        break
                                    pt["text"] = combined_text
                                    pt["cx"] = (pt["cx"] + rt["cx"]) / 2
                                    pt["cy"] = (pt["cy"] + rt["cy"]) / 2
                                    pt["bottom_y"] = max(pt["bottom_y"], rt["bottom_y"])
                                    pt["w"] = max(pt["w"], rt["w"])
                                    pt["h"] = max(pt["h"], rt["h"])
                                    pt["ocr_x_left"] = min(pt["ocr_x_left"], rt["ocr_x_left"])
                                    pt["ocr_x_right"] = max(pt["ocr_x_right"], rt["ocr_x_right"])
                                    merged = True
                                    break
                            if not merged:
                                potential_titles.append(rt)
                                
                        for pt in potential_titles:
                            pt["id"] = global_title_id
                            all_potential_titles.append(pt)
                            global_title_id += 1
                    except Exception as e:
                        print(f"RapidOCR 執行發生錯誤: {e}")
                        pass

            valid_ids_set = set()
            skip_llm = cv_params.get("skip_llm_filter", False)
            if all_potential_titles:
                import json
                print(f"[Phase 3.5.5] 全域收集到 {len(all_potential_titles)} 個標題候選: {[t['text'] for t in all_potential_titles]}")
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key and not skip_llm:
                    try:
                        import google.generativeai as l_genai
                        l_genai.configure(api_key=api_key)
                        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
                        model = l_genai.GenerativeModel(model_name)
                        prompt = """你是一位嚴格的結構工程師。我正在進行台灣建築結構配筋圖（RC梁）的自動切圖作業。
以下是一批由 OCR 從圖紙擷取出來的候選文字，裡面混雜了嚴重的圖面雜訊，請幫我**嚴苛地**剔除雜訊。

【合法梁標題特徵】（必須保留）：
1. 必須是明確的 RC 梁編號與尺寸標示。
2. 格式通常為：「樓層」+「梁代號」+「尺寸」。
   - 例如：B3F B3-4(50x70)、B3F CB5-1(60x80)、1F G2 (40x60)、RF b2-1（30x50)
3. 偶爾沒有樓層，但「梁代號 +(尺寸)」是絕對核心。

【絕對要剔除的雜訊】（絕對不可保留）：
1. 圖號/建檔號：如 'f4915', '134e15', '8fS', '13-4e15-'
2. 網格線座標/區位碼：如 '3-F8', 'S4-01', '3-B'
3. 中文施工說明/附註長句：如 '1.RC柱、梁因现最施工...'、'調整前使社...' 等任何說明文字。
4. 英文公司名/人名：如 'H.P.ChuehArchitects&Planners'
5. 單純的鋼筋號數或數字：如 '4-#8', '2-10'。

請你嚴格遵守上述規則，寧可錯殺也不要放過雜訊！只保留真實的「RC 梁標題」。
請「嚴格且僅以」 JSON 陣列 (List of Integers) 格式輸出合法標題的 id。
例如：如果 id 1, 5 是合法梁標題，請輸出 [1, 5]。不要回傳任何額外字串、不要解釋！(確保格式為純粹的 JSON 陣列，如 ```json [1,2] ```)。

【輸入字串列表】：
""" + json.dumps([{"id": t["id"], "text": t["text"]} for t in all_potential_titles], ensure_ascii=False)
                        response = model.generate_content(prompt)
                        text = response.text
                        s_idx = text.find('[')
                        e_idx = text.rfind(']')
                        if s_idx != -1 and e_idx != -1:
                            valid_ids_list = json.loads(text[s_idx:e_idx+1])
                            valid_ids_set = set(valid_ids_list)
                            valid_titles_text = [t["text"] for t in all_potential_titles if t["id"] in valid_ids_set]
                            print(f"[Phase 3.5.5] LLM 高效雜訊過濾完成: 保留 {len(valid_ids_set)} / {len(all_potential_titles)} 個標題")
                            print(f"[Phase 3.5.5] LLM 回傳採納之梁標題清單: {valid_titles_text}")
                        else:
                            raise ValueError("回傳不在 JSON")
                    except Exception as e:
                        print(f"[Phase 3.5.5] LLM 雜訊過濾失敗 ({e})，退回本地正規表達式嚴格過濾。")
                        # 本地 fallback：必須包含「(數字x數字)」尺寸標記才算合格梁標題
                        for t in all_potential_titles:
                            if re.search(r'\d+\s*[xX×*]\s*\d+', t["text"]):
                                valid_ids_set.add(t["id"])
                        print(f"[Phase 3.5.5] 本地 Regex 過濾: 保留 {len(valid_ids_set)} / {len(all_potential_titles)} 個標題")
                else:
                    if skip_llm:
                        print("[Phase 3.5.5] 已啟用純本地模式，退回本地 Regex 過濾。")
                    elif not api_key:
                        print("[Phase 3.5.5] ⚠️ 未設定 GEMINI_API_KEY，退回本地 Regex 過濾。")
                    for t in all_potential_titles:
                        if re.search(r'\d+\s*[xX×*]\s*\d+', t["text"]):
                            valid_ids_set.add(t["id"])

            confirmed_titles_list = [t for t in all_potential_titles if t["id"] in valid_ids_set]
            
            from core.normalizer import normalize_text
            for ct in confirmed_titles_list:
                ct["text"] = normalize_text(ct["text"])

        # === Phase 3.6: 精準 Title Reclaim (標題歸屬回收) ===
        title_reclaim_count = 0
        claimed_title_ids = set()
        
        for i, bbox in enumerate(results):
            orig_x0, orig_y0, orig_x1, orig_y1 = bbox
            
            # 先檢查母塊內部是否已經有標題存在，如果有就跳過 Reclaim
            has_internal_title = False
            for ct in confirmed_titles_list:
                pdf_cx = ct["cx"] / 4.0
                pdf_cy = ct["cy"] / 4.0
                if orig_x0 - 10 <= pdf_cx <= orig_x1 + 10 and orig_y0 <= pdf_cy <= orig_y1:
                    has_internal_title = True
                    claimed_title_ids.add(ct["id"])
            if has_internal_title:
                continue
            
            owned_titles = []
            for ct in confirmed_titles_list:
                # Is the title within reasonable X range?
                if ct["ocr_x_right"] >= orig_x0 - 60 and ct["ocr_x_left"] <= orig_x1 + 60:
                    # Is it inside or vertically below up to 150pt?
                    pdf_cy = ct["cy"] / 4.0
                    if pdf_cy >= orig_y1 - 30 and pdf_cy <= orig_y1 + 150:
                        owned_titles.append(ct)
                        claimed_title_ids.add(ct["id"])
            
            if owned_titles:
                title_y_max = max(t["bottom_y"] / 4.0 for t in owned_titles) + 5
                if title_y_max > bbox[3]:
                    bbox[3] = title_y_max
                    for t in owned_titles:
                        bbox[0] = min(bbox[0], t["ocr_x_left"] - 10)
                        bbox[2] = max(bbox[2], t["ocr_x_right"] + 10)
                    title_reclaim_count += 1
                    
        # === 幽靈標題復活機制 (Ghost Title Resurrection) ===
        # 對於那些因為 OpenCV 繪圖太小 (area < min_area) 或未閉合而被遺棄，
        # 但卻被 LLM 確認為合法標題的文字，我們主動生成預設的幾何方塊來復活它們。
        resurrected_count = 0
        for ct in confirmed_titles_list:
            if ct["id"] not in claimed_title_ids:
                pdf_cy = ct["cy"] / 4.0
                pdf_cx = ct["cx"] / 4.0
                # 預設梁的位置在標題正上方約 30pt 的位置，高度約 40pt
                rx0 = max(0, ct["ocr_x_left"] - 15)
                ry0 = max(0, pdf_cy - 45)
                rx1 = min(page.rect.width, ct["ocr_x_right"] + 15)
                ry1 = min(page.rect.height, pdf_cy + 10)
                results.append([rx0, ry0, rx1, ry1])
                resurrected_count += 1
                
        if title_reclaim_count > 0 or resurrected_count > 0:
            msg = f"[Phase 3.6] 標題歸屬回收: 延伸 {title_reclaim_count} 個 bbox。"
            if resurrected_count > 0:
                msg += f" 復活了 {resurrected_count} 個遺失實體的合法標題。"
            print(msg)
# === Phase 3.6.5: 二次貪婪融合 (Post-Reclaim NMS) ===
        # 因為標題往下延伸後，極有可能侵犯到下方的其他母塊，
        # 此處再跑一次以 IoA 為基礎的貪婪融合，將重疊區塊合體。
        original_len = len(results)
        results, post_nms_drops = self._nms_bboxes(results, iou_thresh=0.4)
        for nd in post_nms_drops:
            dropped_for_save.append(("nms_post_reclaim", nd))
            
        if len(results) < original_len:
            print(f"[Phase 3.6.5] 二次貪婪融合: 因標題向下擴張產生重疊，將 {original_len} 個母塊合體為 {len(results)} 個")
        
        # === Phase 3.6.6: 頁面邊界夾緊 + 內容邊界收緊 (母塊級) ===
        # 母塊階段：不 trim 底部 (留給 Phase 3.7 標題截斷處理)
        pw, ph = page.rect.width, page.rect.height
        self._content_trim_bboxes(results, thresh, pw, ph,
                                  pad_x=60, pad_y=20, trim_bottom=False)
        
        # === Phase 3.6.7: 第三輪 NMS (Post-Trim Dedup) ===
        # Content Trim 收緊邊界後，原本因雜訊向不同方向膨脹（一張左右、一張上下）
        # 而未觸發融合的重疊母塊，此時 IoA 會大幅提高，可以被正確合併。
        pre_trim_nms_len = len(results)
        results, post_trim_drops = self._nms_bboxes(results, iou_thresh=0.4)
        for nd in post_trim_drops:
            dropped_for_save.append(("nms_post_trim", nd))
        if len(results) < pre_trim_nms_len:
            print(f"[Phase 3.6.7] Post-Trim NMS: 收緊後偵測到重疊，{pre_trim_nms_len} → {len(results)} 個母塊")
        # ==============================================================

                # === Phase 3.8: 連續跨水平分解 (Continuous Beam Decomposition) ===
        final_single_spans = []
        original_parents = []
        child_to_parent_map = {}
        
        if enable_decomp:
            trimmed_parent_logs = []
            phase37_deleted = 0
            phase37_split = 0
            
            for p_idx, bbox in enumerate(results):
                orig_x0, orig_y0, orig_x1, orig_y1 = bbox
                
                # Find which confirmed titles fell into this merged mother block
                filtered_titles = []
                for ct in confirmed_titles_list:
                    pdf_cx = ct["cx"] / 4.0
                    pdf_cy = ct["cy"] / 4.0
                    # 恢復原本的容錯光暈 (±20pt)。因為我們現在有強大的重疊回收，不怕蹭飯塊，但怕漏標題！
                    if pdf_cx >= orig_x0 - 20 and pdf_cx <= orig_x1 + 20 and pdf_cy >= orig_y0 - 20 and pdf_cy <= orig_y1 + 20:
                        filtered_titles.append(ct)
                        
                # === Phase 3.7 自檢 規則 1: 無合法標題 → 刪除 ===
                if len(filtered_titles) == 0:
                    phase37_deleted += 1
                    print(f"[Phase 3.7] 刪除無標題母塊 {p_idx} (y0={orig_y0:.1f})")
                    # No longer appending to original_parents!
                    continue
                    
                # === Phase 3.7 自檢 規則 2: Y 軸多排偵測 ===
                sorted_by_y = sorted(filtered_titles, key=lambda t: t["cy"])
                y_groups = [[sorted_by_y[0]]]
                for t in sorted_by_y[1:]:
                    if t["cy"] - y_groups[-1][-1]["cy"] > 60: # 60px at 4x = 15pt
                        y_groups.append([t])
                    else:
                        y_groups[-1].append(t)

                # 檢查相鄰群的 X 範圍是否有重疊，沒重疊代表是左右排列而非上下疊放
                has_x_overlap = False
                if len(y_groups) >= 2:
                    for gi in range(len(y_groups) - 1):
                        g_a = y_groups[gi]
                        g_b = y_groups[gi + 1]
                        a_left  = min(t["ocr_x_left"]  for t in g_a)
                        a_right = max(t["ocr_x_right"] for t in g_a)
                        b_left  = min(t["ocr_x_left"]  for t in g_b)
                        b_right = max(t["ocr_x_right"] for t in g_b)
                        if a_left < b_right and b_left < a_right:
                            has_x_overlap = True
                            break

                if len(y_groups) >= 2 and has_x_overlap:
                    phase37_split += 1
                    print(f"[Phase 3.7] 母塊 {p_idx} 垂直分割: {len(y_groups)} 排, 標題: {[t['text'] for t in filtered_titles]}")
                    
                    prev_pdf_y = orig_y0
                    for g_idx, group in enumerate(y_groups):
                        lowest_bottom = max(t["bottom_y"] for t in group)
                        
                        pdf_lowest_bottom = lowest_bottom / 4.0
                        sub_y1 = min(orig_y1, pdf_lowest_bottom + 5)
                        sub_y1 = max(sub_y1, prev_pdf_y + 1)
                        
                        if g_idx < len(y_groups) - 1:
                            cut_pdf_y = pdf_lowest_bottom + 20 / 4.0 # 5pt below
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, sub_y1]
                            original_parents.append(sub_bbox)
                            trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": group})
                            prev_pdf_y = max(cut_pdf_y, sub_y1)
                        else:
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, sub_y1]
                            original_parents.append(sub_bbox)
                            trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": group})
                    continue
                    
                # 單排母塊
                lowest_y_pdf = max(t["bottom_y"] for t in filtered_titles) / 4.0
                new_orig_y1 = min(orig_y1, lowest_y_pdf + 5)
                new_orig_y1 = max(new_orig_y1, orig_y0 + 1)
                
                original_parents.append([orig_x0, orig_y0, orig_x1, new_orig_y1])
                trimmed_parent_logs.append({"idx": len(original_parents) - 1, "titles": filtered_titles})
                
            if phase37_deleted > 0 or phase37_split > 0:
                print(f"[Phase 3.7] Y軸自檢: 刪除 {phase37_deleted} 無標題塊, 垂直分割 {phase37_split} 多排塊")
                
            # === Final Deduplication (因為切塊後可能有 100% 重疊的切片) ===
            dedup_parents = []
            dedup_logs = []
            for i, p_bbox in enumerate(original_parents):
                merged = False
                for j, kp_bbox in enumerate(dedup_parents):
                    inter_x0 = max(p_bbox[0], kp_bbox[0])
                    inter_y0 = max(p_bbox[1], kp_bbox[1])
                    inter_x1 = min(p_bbox[2], kp_bbox[2])
                    inter_y1 = min(p_bbox[3], kp_bbox[3])
                    inter_area = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
                    area_small = min((p_bbox[2]-p_bbox[0])*(p_bbox[3]-p_bbox[1]), (kp_bbox[2]-kp_bbox[0])*(kp_bbox[3]-kp_bbox[1]))
                    
                    if inter_area / (area_small + 1e-6) > 0.6: # 60% 重疊就視為同一個
                        # 融合
                        kp_bbox[0] = min(kp_bbox[0], p_bbox[0])
                        kp_bbox[1] = min(kp_bbox[1], p_bbox[1])
                        kp_bbox[2] = max(kp_bbox[2], p_bbox[2])
                        kp_bbox[3] = max(kp_bbox[3], p_bbox[3])
                        # 標題聯集
                        existing_titles = {t["id"] for t in dedup_logs[j]["titles"]}
                        for t in trimmed_parent_logs[i]["titles"]:
                            if t["id"] not in existing_titles:
                                dedup_logs[j]["titles"].append(t)
                                existing_titles.add(t["id"])
                        merged = True
                        break
                if not merged:
                    dedup_parents.append(p_bbox)
                    dedup_logs.append(trimmed_parent_logs[i])
                    
            if len(original_parents) > len(dedup_parents):
                print(f"[Phase 3.8] 最終重疊回收: 發現 {len(original_parents) - len(dedup_parents)} 塊切片 100% 疊代，已完美融合。")
                
            original_parents = dedup_parents
            trimmed_parent_logs = dedup_logs
            
            # === Phase 3.7.2: 融合後二次垂直分割 (Final Y-Axis split) ===
            # 因為上方的最終重疊回收，可能會把兩塊原本只含單排的圖塊融合成一塊含多排標題的大圖塊
            # 這裡再執行一次水平斬波 (垂直分割)，將它們依標題排數精準切開
            final_parents = []
            final_logs = []
            final_split_count = 0
            for i, p_bbox in enumerate(original_parents):
                orig_x0, orig_y0, orig_x1, orig_y1 = p_bbox
                filtered_titles = trimmed_parent_logs[i]["titles"]
                if len(filtered_titles) == 0:
                     continue
                     
                sorted_by_y = sorted(filtered_titles, key=lambda t: t["cy"])
                y_groups = [[sorted_by_y[0]]]
                for t in sorted_by_y[1:]:
                    if t["cy"] - y_groups[-1][-1]["cy"] > 60: # 60px at 4x = 15pt
                        y_groups.append([t])
                    else:
                        y_groups[-1].append(t)
                        
                # 檢查相鄰群的 X 範圍是否有重疊
                has_x_overlap = False
                if len(y_groups) >= 2:
                    for gi in range(len(y_groups) - 1):
                        g_a = y_groups[gi]
                        g_b = y_groups[gi + 1]
                        a_left  = min(t["ocr_x_left"]  for t in g_a)
                        a_right = max(t["ocr_x_right"] for t in g_a)
                        b_left  = min(t["ocr_x_left"]  for t in g_b)
                        b_right = max(t["ocr_x_right"] for t in g_b)
                        if a_left < b_right and b_left < a_right:
                            has_x_overlap = True
                            break

                if len(y_groups) >= 2 and has_x_overlap:
                    final_split_count += 1
                    print(f"[Phase 3.7.2] 融合後二次水平斬波: 將融合大塊切成 {len(y_groups)} 排")
                    
                    prev_pdf_y = orig_y0
                    for g_idx, group in enumerate(y_groups):
                        lowest_bottom = max(t["bottom_y"] for t in group)
                        
                        pdf_lowest_bottom = lowest_bottom / 4.0
                        sub_y1 = min(orig_y1, pdf_lowest_bottom + 5)
                        sub_y1 = max(sub_y1, prev_pdf_y + 1)
                        
                        if g_idx < len(y_groups) - 1:
                            cut_pdf_y = pdf_lowest_bottom + 20 / 4.0
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, sub_y1]
                            final_parents.append(sub_bbox)
                            final_logs.append({"idx": len(final_parents)-1, "titles": group})
                            prev_pdf_y = max(cut_pdf_y, sub_y1)
                        else:
                            sub_bbox = [orig_x0, prev_pdf_y, orig_x1, sub_y1]
                            final_parents.append(sub_bbox)
                            final_logs.append({"idx": len(final_parents)-1, "titles": group})
                    continue
                
                # 單排不切
                final_parents.append(p_bbox)
                final_logs.append(trimmed_parent_logs[i])
                
            if final_split_count > 0:
                print(f"[Phase 3.7.2] 成功對 {final_split_count} 個剛剛融合完成的跨排母塊執行二次水平斬波。")
                
            original_parents = final_parents
            trimmed_parent_logs = final_logs
            
            # === Phase 3.9: Final Sanity Check ===
            # 刪除被壓縮成一條線 (<2px) 的殘骸，或是文字物理上根本不在框內的幽靈圖塊
            valid_parents = []
            valid_logs = []
            for i, p_bbox in enumerate(original_parents):
                orig_x0, orig_y0, orig_x1, orig_y1 = p_bbox
                if orig_y1 - orig_y0 <= 2.0:
                    continue  # 高度 <= 2 的絕對是雜區線條殘骸，直接刪除
                    
                valid_titles = []
                for ct in trimmed_parent_logs[i]["titles"]:
                    pdf_cx = ct["cx"] / 4.0
                    pdf_cy = ct["cy"] / 4.0
                    # 容許 ±10pt 的光暈，但如果差太遠 (例如文字在框外 20pt) 就剔除
                    if orig_x0 - 10 <= pdf_cx <= orig_x1 + 10 and orig_y0 - 10 <= pdf_cy <= orig_y1 + 10:
                        valid_titles.append(ct)
                        
                if not valid_titles:
                    continue # 如果剔除後沒半個標題，這個母塊也刪除
                    
                trimmed_parent_logs[i]["titles"] = valid_titles
                valid_parents.append(p_bbox)
                valid_logs.append(trimmed_parent_logs[i])
                
            original_parents = valid_parents
            trimmed_parent_logs = valid_logs
            
            final_single_spans = list(original_parents)
            child_to_parent_map = {i: i for i in range(len(final_single_spans))}
            print(f"[Phase 3.8] 原有 {len(final_parents)} 母塊。已依指示關閉 X 軸斬波，共輸出 {len(final_single_spans)} 個單純受 Y 軸修整之母圖塊。")
            results = final_single_spans            
            # === Pass 1 Content Trim (單跨級) ===
            # 每個單跨各自依內容收緊，底部也 trim (Phase 3.7 已截斷過了)
            self._content_trim_bboxes(results, thresh, pw, ph,
                                      pad_x=40, pad_y=20, trim_bottom=True)
        # ==============================================================
        # 執行除錯裁切並儲存
        if dropped_for_save or final_single_spans or original_parents:
            
            drop_dir = "crops/drop"
            trimmed_dir = "crops/trimmed_parents"
            os.makedirs(drop_dir, exist_ok=True)
            os.makedirs(trimmed_dir, exist_ok=True)
            mat_save = fitz.Matrix(2.0, 2.0)
            
            for idx, (reason, rect_coords) in enumerate(dropped_for_save):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    img.save(os.path.join(drop_dir, f"drop_{reason}_{idx}.png"))
                except Exception as e:
                    pass
                    
            for idx, rect_coords in enumerate(original_parents):
                try:
                    r = fitz.Rect(rect_coords)
                    r = r.intersect(page.rect)
                    if r.is_empty: continue
                    pix_drop = page.get_pixmap(matrix=mat_save, clip=r)
                    img = Image.open(io.BytesIO(pix_drop.tobytes("png")))
                    # 這是被 Phase 3.5 垂直微聚類過濾後的原始母圖
                    img.save(os.path.join(trimmed_dir, f"trimmed_parent_{idx}.png"))
                except Exception as e:
                    pass
                    
            try:
                log_path = os.path.join(trimmed_dir, "titles_summary.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    for log in trimmed_parent_logs:
                        f.write(f"▼ 母塊 trimmed_parent_{log['idx']}.png\n")
                        f.write(f"  包含 {len(log['titles'])} 個過濾後的真梁編號:\n")
                        for t in log["titles"]:
                            f.write(f"    - {t['text']} (X={t['cx']:.1f}, Y={t['cy']:.1f})\n")
                        f.write("\n")
            except Exception as e:
                print("Failed to write titles_summary.txt", e)
        
        metrics = {
            "total_contours": total_contours,
            "noise_dropped": noise_dropped,
            "noise_drop_rate": round((noise_dropped / total_contours) * 100, 1) if total_contours > 0 else 0,
            "nms_dropped": int(nms_dropped),
            "nms_drop_rate": round((nms_dropped / pre_nms_len) * 100, 1) if pre_nms_len > 0 else 0,
            "parent_count": len(original_parents),
            "child_count": len(final_single_spans),
            "child_to_parent_map": child_to_parent_map,
            "original_parents": original_parents,
            "trimmed_parent_logs": trimmed_parent_logs,
            "final_single_spans": final_single_spans,
            "_thresh": thresh,  # 二值圖供下游 Pass 2 content trim 使用
            "_page_w": pw,
            "_page_h": ph
        }
        
        return results, metrics
