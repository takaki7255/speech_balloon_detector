#!/usr/bin/env python3
"""
test_manga_processor.py
=======================

manga_processor.pyのテスト用スクリプト
単一画像での動作確認用
"""

import cv2
import numpy as np
from pathlib import Path
from manga_processor import MangaProcessor

def test_single_image():
    """単一画像でのテスト"""
    
    # テスト画像パス（適宜変更してください）
    test_image_path = "./../manga_109_all/collected_images/000325.jpg"
    output_dir = "./test_output"
    
    # 出力ディレクトリ作成
    Path(output_dir).mkdir(exist_ok=True)
    
    # 画像読み込み（manga_processor.pyと同じグレースケール読み込み）
    img = cv2.imread(test_image_path, 0)  # 0でグレースケール読み込み
    if img is None:
        print(f"Error: Cannot load image from {test_image_path}")
        return
    
    print(f"Loaded image: {test_image_path}")
    print(f"Image shape: {img.shape}")
    
    # プロセッサー初期化
    processor = MangaProcessor("dummy", output_dir)
    
    # ページ分割テスト
    print("\n=== Page Cut Test ===")
    pages = processor.page_cut(img)
    print(f"Split into {len(pages)} pages")
    
    for i, page in enumerate(pages):
        print(f"Page {i}: {page.shape}")
        cv2.imwrite(f"{output_dir}/page_{i}.png", page)
    
    # 各ページを処理
    for page_idx, page in enumerate(pages):
        print(f"\n=== Processing Page {page_idx} ===")
        
        # ページ分類テスト
        is_black = processor.get_page_type(page)
        print(f"Page type: {'Black' if is_black else 'White'}")
        
        if is_black:
            print("Skipping black page")
            continue
        
        # フレーム検出テスト
        print("Detecting frames...")
        panels = processor.frame_detect(page)
        print(f"Detected {len(panels)} panels")
        
        # パネル保存
        for panel_idx, panel in enumerate(panels):
            panel_filename = f"{output_dir}/panel_{page_idx}_{panel_idx}.png"
            cv2.imwrite(panel_filename, panel.image)
            print(f"Saved panel: {panel_filename}")
            
            # 吹き出し検出テスト
            print(f"  Detecting balloons in panel {panel_idx}...")
            balloons = processor.speechballoon_detect(panel.image)
            print(f"  Found {len(balloons)} balloon candidates")
            
            # 誤検出除去テスト
            filtered_balloons = processor.remove_false_balloons(balloons)
            print(f"  After filtering: {len(filtered_balloons)} balloons")
            
            # 吹き出し保存
            for balloon_idx, balloon in enumerate(filtered_balloons):
                balloon_filename = f"{output_dir}/balloon_{page_idx}_{panel_idx}_{balloon_idx}.png"
                cv2.imwrite(balloon_filename, balloon.image)
                print(f"    Saved balloon: {balloon_filename}")
                print(f"    - Type: {['Circle', 'Rect', 'Zigzag'][balloon.type]}")
                print(f"    - Area: {balloon.area:.0f}")
                print(f"    - Circularity: {balloon.circularity:.3f}")
                print(f"    - B/W ratio: {balloon.bw_ratio:.3f}")
    
    print(f"\nTest completed! Check output in: {output_dir}")


def visualize_detection_steps():
    """検出ステップの可視化"""
    
    test_image_path = "./../manga_109_all/collected_images/000325.jpg"
    output_dir = "./debug_output"
    Path(output_dir).mkdir(exist_ok=True)
    
    img = cv2.imread(test_image_path, 0)  # グレースケール読み込み
    if img is None:
        print(f"Error: Cannot load image from {test_image_path}")
        return
    
    processor = MangaProcessor("dummy", output_dir)
    pages = processor.page_cut(img)
    
    for page_idx, page in enumerate(pages):
        if processor.get_page_type(page):
            continue
            
        print(f"\n=== Debug Page {page_idx} ===")
        
        # グレースケール
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_1_gray.png", gray)
        
        # 二値化
        _, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_2_binary.png", bin_img)
        
        # モルフォロジー
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.erode(bin_img, kernel, iterations=1)
        morph = cv2.dilate(morph, kernel, iterations=1)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_3_morph.png", morph)
        
        # ガウシアンフィルタ
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_4_gaussian.png", gaussian)
        
        # 二値化（反転）
        _, inverse_bin = cv2.threshold(gaussian, 210, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_5_inverse_binary.png", inverse_bin)
        
        # Canny
        canny = cv2.Canny(gray, 120, 130, 3)
        cv2.imwrite(f"{output_dir}/debug_{page_idx}_6_canny.png", canny)
        
        print(f"Debug images saved for page {page_idx}")
        break  # 最初のページのみ


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        visualize_detection_steps()
    else:
        test_single_image()