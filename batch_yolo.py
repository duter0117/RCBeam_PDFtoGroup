import argparse
import os
import json
import fitz
from PIL import Image, ImageDraw, ImageFont
import glob
from dotenv import load_dotenv
import sys
import warnings

# 強制關閉輸出緩衝，讓 print 立刻顯示在終端機
sys.stdout.reconfigure(line_buffering=True)
# 隱藏 Google API 的 FutureWarning 以免誤導為報錯
warnings.filterwarnings("ignore", category=FutureWarning)
from core.vector_extractor import VectorExtractor

load_dotenv()  # 載入 .env 裡面的 GEMINI_API_KEY

def process_single_pdf(pdf_path, out_dir, page_num=0):
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]
    
    cv_params = {
        "dilation_iterations": 2,
        "min_area": 3000,
        "padding_bottom": 1,
        "hough_threshold": 95,
        "enable_decomp": True,
        "skip_llm_filter": False, # 預設開啟 LLM 篩選雜訊
    }
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    print(f"[{base_name}] 開始分析 (Page {page_num})...")
    with VectorExtractor(pdf_bytes) as extractor:
        # Step 1: 分析 Bounding Boxes
        _, metrics = extractor.extract_opencv_bboxes(page_num, cv_params)
        
        page_w = metrics.get("_page_w", 0)
        page_h = metrics.get("_page_h", 0)
        
        if page_w == 0 or page_h == 0:
            print(f"[{base_name}] 錯誤: 無法獲取圖紙尺寸。跳過。")
            return
            
        continuous_beams_data = []
        original_parents = metrics.get("original_parents", [])
        child_to_parent_map = metrics.get("child_to_parent_map", {})
        final_single_spans = metrics.get("final_single_spans", [])
        trimmed_parent_logs = metrics.get("trimmed_parent_logs", [])
        
        parent_to_spans = {}
        for dict_key, p_idx in child_to_parent_map.items():
            if str(p_idx) not in parent_to_spans:
                parent_to_spans[str(p_idx)] = []
            try:
                real_c_idx = int(dict_key)
                if real_c_idx < len(final_single_spans):
                    parent_to_spans[str(p_idx)].append(final_single_spans[real_c_idx])
            except ValueError:
                pass
            
        for idx, parent_bbox in enumerate(original_parents):
            titles_info = []
            for log in trimmed_parent_logs:
                if log["idx"] == idx:
                    for t in log["titles"]:
                        pw = t.get("w", 0) / 4.0
                        ph = t.get("h", 0) / 4.0
                        pcx = t.get("cx", 0) / 4.0
                        pcy = t.get("cy", 0) / 4.0
                        t_x0 = pcx - pw/2.0
                        t_y0 = pcy - ph/2.0
                        t_x1 = pcx + pw/2.0
                        t_y1 = pcy + ph/2.0
                        titles_info.append({
                            "text": t.get("text", ""),
                            "bbox": [round(t_x0, 2), round(t_y0, 2), round(t_x1, 2), round(t_y1, 2)]
                        })
                    break
                    
            spans = parent_to_spans.get(str(idx), [])
            if not spans:
                spans = [parent_bbox]
                
            continuous_beams_data.append({
                "parent_id": idx,
                "parent_bbox": [round(x, 2) for x in parent_bbox],
                "titles": titles_info,
                "single_spans": [[round(x, 2) for x in s] for s in spans]
            })
            
        # 準備資料與 YOLO 格式資料夾架構
        img_dir = os.path.join(out_dir, "images")
        lbl_dir = os.path.join(out_dir, "labels")
        dbg_dir = os.path.join(out_dir, "debug_images")
        jsn_dir = os.path.join(out_dir, "json_exports")
        
        for d in [img_dir, lbl_dir, dbg_dir, jsn_dir]:
            os.makedirs(d, exist_ok=True)
            
        json_path = os.path.join(jsn_dir, f"{base_name}.json")
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        img_path = os.path.join(img_dir, f"{base_name}.png")
        dbg_path = os.path.join(dbg_dir, f"{base_name}_debug.png")
        
        # Step 2: 輸出 JSON
        yolo_data = {
            "page_info": {"page_num": page_num, "width": round(page_w, 2), "height": round(page_h, 2)},
            "continuous_beams": continuous_beams_data
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(yolo_data, f, ensure_ascii=False, indent=2)
            
        # Step 3: 輸出 YOLO txt
        labels_output = []
        def to_yolo(bbox_pdf, class_id):
            x0, y0, x1, y1 = bbox_pdf
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            bw, bh = (x1 - x0), (y1 - y0)
            norm_x, norm_y = cx / page_w, cy / page_h
            norm_w, norm_h = bw / page_w, bh / page_h
            norm_x, norm_y = max(0.0, min(1.0, norm_x)), max(0.0, min(1.0, norm_y))
            norm_w, norm_h = max(0.0, min(1.0, norm_w)), max(0.0, min(1.0, norm_h))
            return f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"

        CLASS_BEAM = 0
        for beam in yolo_data["continuous_beams"]:
            labels_output.append(to_yolo(beam["parent_bbox"], CLASS_BEAM))

        with open(lbl_path, "w") as f:
            f.write("\n".join(labels_output))
            
        # Step 4: 輸出乾淨的訓練原圖 (YOLO Train Image) 與 Debug 畫框圖片
        # 為了平衡檔案大小與解析度，我們用 fitz.Matrix(2.0, 2.0) (大約 144 DPI)
        scale_factor = 2.0
        doc = fitz.open("pdf", pdf_bytes)
        mat = fitz.Matrix(scale_factor, scale_factor)
        pix = doc[page_num].get_pixmap(matrix=mat)
        
        # 1. 存給 YOLO 用的乾淨圖片
        pix.save(img_path)
        
        # 2. 畫上檢驗框框存成 Debug 圖片
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for beam in yolo_data["continuous_beams"]:
            # 畫「連續跨母梁」紅框 (粗度改細)
            px0, py0, px1, py1 = [coord * scale_factor for coord in beam["parent_bbox"]]
            draw.rectangle([px0, py0, px1, py1], outline="red", width=2)
            
            # 加上編號文字
            draw.text((px0 + 4, py0 + 4), f"Parent {beam['parent_id']}", fill="red")
            
            # 依據你的要求，不再繪製「綠色子標題」與「藍色單跨」框
                
        img.save(dbg_path)
        print(f"[{base_name}] 完成！(連續跨數: {len(continuous_beams_data)})")


def main():
    parser = argparse.ArgumentParser(description="批次擷取 PDF 並轉出 YOLO 訓練集 (含 Debug 圖片)")
    parser.add_argument("input_path", type=str, help="PDF 檔案路徑 或 包含多個 PDF 的資料夾路徑")
    parser.add_argument("--page", type=int, default=0, help="要解析的頁碼 (預設第 0 頁)")
    parser.add_argument("--out_dir", type=str, default="yolo_dataset", help="輸出資料夾的根目錄 (預設 'yolo_dataset')")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"[錯誤] 找不到輸入路徑: {args.input_path}")
        return
        
    pdf_files = []
    if os.path.isdir(args.input_path):
        # 批次模式，尋找資料夾下所有 pdf
        pdf_files = glob.glob(os.path.join(args.input_path, "*.pdf"))
    elif args.input_path.lower().endswith('.pdf'):
        # 單檔模式
        pdf_files = [args.input_path]
    else:
        print("[錯誤] 輸入的檔案不是 PDF")
        return
        
    if not pdf_files:
        print("[提示] 找不到任何 PDF 檔案")
        return
        
    print(f"即將批次處理 {len(pdf_files)} 份 PDF 檔案...")
    
    for pdf in pdf_files:
        try:
            process_single_pdf(pdf, args.out_dir, args.page)
        except Exception as e:
            print(f"[錯誤] 處理 {os.path.basename(pdf)} 時發生例外: {e}")
            
    print(f"\n全部執行完畢！資料已匯出至 {os.path.abspath(args.out_dir)}")
    print("目錄結構如下：")
    print(f" - {args.out_dir}/images/       # 乾淨的原始圖片 (給 YOLO 訓練用)")
    print(f" - {args.out_dir}/labels/       # YOLO 座標標籤檔案 (.txt)")
    print(f" - {args.out_dir}/debug_images/ # 給你人工檢驗標註品質的畫框圖片 (紅色=連續跨母梁)")
    print(f" - {args.out_dir}/json_exports/ # 原生的座標 JSON 檔")

if __name__ == "__main__":
    main()
