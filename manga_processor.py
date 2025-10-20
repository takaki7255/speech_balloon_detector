#!/usr/bin/env python3
"""
manga_processor.py
==================

C++のmain.cppとその関連ファイル群をPythonで再現した統合版
- ページ分割（2ページ → 1ページ）
- ページ分類（白ページ/黒ページ）
- フレーム（コマ）検出・抽出
- 吹き出し検出・抽出

Usage:
    python manga_processor.py <input_folder> <output_folder>
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import glob


@dataclass
class Point:
    """座標点を表すクラス"""
    x: int
    y: int


@dataclass
class Points:
    """矩形の4つの角を表すクラス"""
    lt: Point  # left-top
    rt: Point  # right-top
    lb: Point  # left-bottom
    rb: Point  # right-bottom


@dataclass 
class Panel:
    """コマ（フレーム）情報"""
    image: np.ndarray  # RGBA画像
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    corners: Points
    page_idx: int
    panel_idx: int


@dataclass
class Balloon:
    """吹き出し情報"""
    image: np.ndarray  # RGBA画像
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    circularity: float
    type: int  # 0:円形, 1:矩形, 2:ギザギザ
    bw_ratio: float
    panel_idx: int


class MangaProcessor:
    """マンガ処理メインクラス"""
    
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # 出力フォルダ構造
        self.panels_dir = self.output_folder / "panels"
        self.balloons_dir = self.output_folder / "balloons"
        self.panels_dir.mkdir(exist_ok=True)
        self.balloons_dir.mkdir(exist_ok=True)
        
        # パラメータ設定
        self.speechballoon_max = 99  # 最大吹き出し数
        
    def get_image_paths(self, extension: str = "jpg") -> List[str]:
        """フォルダから画像パスを取得"""
        pattern = str(self.input_folder / f"*.{extension}")
        paths = glob.glob(pattern)
        paths.sort()
        return paths
    
    def page_cut(self, img: np.ndarray) -> List[np.ndarray]:
        """
        2ページ画像を1ページずつに分割
        C++の PageCut::pageCut() に相当
        """
        h, w = img.shape[:2]
        
        # 単純に左右に分割（実際はより複雑な処理が必要）
        left_page = img[:, :w//2]
        right_page = img[:, w//2:]
        
        return [left_page, right_page]
    
    def get_page_type(self, page: np.ndarray) -> bool:
        """
        ページ分類：白ページ(False) / 黒ページ(True)
        C++の ClassificationPage::get_page_type() に相当
        """
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if len(page.shape) == 3 else page
        mean_intensity = np.mean(gray)
        
        # 平均輝度で判定（閾値は調整が必要）
        return mean_intensity < 100  # 100未満なら黒ページ
    
    def frame_detect(self, page: np.ndarray) -> List[Panel]:
        """
        フレーム（コマ）検出・抽出
        C++の Framedetect::frame_detect() に相当
        """
        panels = []
        
        # グレースケール変換
        if len(page.shape) == 3:
            gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        else:
            gray = page
            
        # 吹き出し検出用の二値化
        _, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        bin_img = cv2.erode(bin_img, np.ones((3,3), np.uint8), iterations=1)
        bin_img = cv2.dilate(bin_img, np.ones((3,3), np.uint8), iterations=1)
        
        # 吹き出し輪郭検出
        contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 吹き出し除去（塗りつぶし）
        self.extract_speech_balloon(contours, gray)
        
        # ガウシアンフィルタ
        gaussian_img = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 二値化（反転）
        _, inverse_bin = cv2.threshold(gaussian_img, 210, 255, cv2.THRESH_BINARY_INV)
        
        # Cannyエッジ検出
        canny = cv2.Canny(gray, 120, 130, 3)
        
        # Hough直線検出
        lines = cv2.HoughLines(canny, 1, np.pi/180, 50)
        
        # 直線画像作成
        lines_img = np.zeros_like(gray)
        if lines is not None:
            for line in lines[:100]:  # 最大100本
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 - 2000 * b)
                y1 = int(y0 + 2000 * a)
                x2 = int(x0 + 2000 * b)
                y2 = int(y0 - 2000 * a)
                cv2.line(lines_img, (x1, y1), (x2, y2), 255, 1)
        
        # 論理積
        and_img = cv2.bitwise_and(inverse_bin, lines_img)
        
        # 輪郭検出
        contours, _ = cv2.findContours(and_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # バウンディングボックスで補完
        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            if self.judge_area_of_bounding_box(bbox, page.shape[0] * page.shape[1]):
                cv2.rectangle(and_img, bbox, 255, 3)
        
        # 最終的な補完画像
        complement_img = cv2.bitwise_and(and_img, inverse_bin)
        
        # 最終輪郭検出
        final_contours, _ = cv2.findContours(complement_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # パネル抽出
        for i, cnt in enumerate(final_contours):
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            
            if not self.judge_area_of_bounding_box(bbox, page.shape[0] * page.shape[1]):
                continue
                
            # 端に寄せる処理
            if x < 6: x = 0
            if y < 6: y = 0
            if x + w > page.shape[1] - 6: w = page.shape[1] - x
            if y + h > page.shape[0] - 6: h = page.shape[0] - y
            
            # RGBA変換
            if len(page.shape) == 3:
                rgba_page = cv2.cvtColor(page, cv2.COLOR_BGR2BGRA)
            else:
                rgba_page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGRA)
            
            # マスク作成
            mask = np.zeros(page.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # 透明化処理
            alpha = rgba_page[:, :, 3]
            alpha[mask == 0] = 0
            
            # 切り出し
            panel_img = rgba_page[y:y+h, x:x+w].copy()
            
            # 角の座標（簡易版）
            corners = Points(
                Point(x, y), Point(x+w, y),
                Point(x, y+h), Point(x+w, y+h)
            )
            
            panel = Panel(
                image=panel_img,
                bbox=(x, y, w, h),
                corners=corners,
                page_idx=0,  # 後で設定
                panel_idx=i
            )
            panels.append(panel)
        
        return panels
    
    def extract_speech_balloon(self, contours: List[np.ndarray], img: np.ndarray):
        """
        吹き出し検出・塗りつぶし（フレーム検出の前処理用）
        C++の Framedetect::extractSpeechBalloon() に相当
        """
        img_area = img.shape[0] * img.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            
            if img_area * 0.008 <= area < img_area * 0.03:
                en = 4.0 * np.pi * area / (peri * peri + 1e-7)
                if en > 0.4:
                    cv2.drawContours(img, [cnt], -1, 0, -1)
    
    def judge_area_of_bounding_box(self, bbox: Tuple[int, int, int, int], page_area: int) -> bool:
        """
        バウンディングボックス面積判定
        C++の Framedetect::judgeAreaOfBoundingBox() に相当
        """
        x, y, w, h = bbox
        return w * h >= 0.048 * page_area
    
    def speechballoon_detect(self, panel: np.ndarray) -> List[Balloon]:
        """
        吹き出し検出・抽出
        C++の Speechballoon::speechballoon_detect() に相当
        """
        balloons = []
        
        # グレースケール変換
        if len(panel.shape) == 4:  # BGRA
            gray = cv2.cvtColor(panel, cv2.COLOR_BGRA2GRAY)
        elif len(panel.shape) == 3:  # BGR
            gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        else:
            gray = panel
        
        # 二値化
        _, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        
        # モルフォロジー処理
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.erode(bin_img, kernel, iterations=1)
        bin_img = cv2.dilate(bin_img, kernel, iterations=1)
        
        # 輪郭検出
        contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        panel_area = panel.shape[0] * panel.shape[1]
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            
            if peri == 0:
                continue
                
            en = 4.0 * np.pi * area / (peri * peri)  # 円形度
            
            # 面積と円形度フィルタ
            if not (panel_area * 0.01 <= area < panel_area * 0.9 and en > 0.4):
                continue
            
            # バウンディングボックス取得
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            # マスク作成
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 0, 4)
            cv2.drawContours(mask, [cnt], -1, 0, -1)
            
            # 背景グレー化
            back_150 = np.full(gray.shape, 150, dtype=np.uint8)
            masked_img = np.where(mask == 0, back_150, gray)
            
            # 切り出し
            cropped_region = masked_img[y:y+h, x:x+w]
            
            # 白黒比率計算
            TH = 255 // 3  # 85
            B = np.sum((cropped_region < TH) & (cropped_region != 150))
            W = np.sum(cropped_region > (255 - TH))
            
            if W == 0 or not (0.01 < (B / W) < 0.7) or B < 10:
                continue
            
            # 形状判定
            max_rect = w * h * 0.95
            if area >= max_rect:
                set_type = 1  # 矩形
            elif en >= 0.7:
                set_type = 0  # 円形
            else:
                set_type = 2  # ギザギザ
            
            # コマサイズ判定
            if w * h >= panel_area * 0.9:
                continue
            
            # RGBA画像作成
            if len(panel.shape) == 3:
                rgba_panel = cv2.cvtColor(panel, cv2.COLOR_BGR2BGRA)
            else:
                rgba_panel = panel.copy()
            
            # 透明化
            alpha_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(alpha_mask, [cnt], -1, 255, -1)
            rgba_panel[:, :, 3] = alpha_mask
            
            # 切り出し
            balloon_img = rgba_panel[y:y+h, x:x+w].copy()
            
            balloon = Balloon(
                image=balloon_img,
                bbox=(x, y, w, h),
                contour=cnt,
                center=(center_x, center_y),
                area=area,
                circularity=en,
                type=set_type,
                bw_ratio=B / W if W > 0 else 0,
                panel_idx=0  # 後で設定
            )
            balloons.append(balloon)
        
        return balloons
    
    def remove_false_balloons(self, balloons: List[Balloon]) -> List[Balloon]:
        """
        誤検出除去処理
        C++の誤検出除去ロジックに相当
        """
        balloon_px_th = 5
        filtered_balloons = []
        
        for balloon in balloons:
            balloon_img = balloon.image
            h, w = balloon_img.shape[:2]
            
            # BGRAをグレースケールに変換
            gray_balloon = cv2.cvtColor(balloon_img, cv2.COLOR_BGRA2GRAY)
            
            # 二値化
            _, bin_balloon = cv2.threshold(gray_balloon, 150, 255, cv2.THRESH_BINARY)
            bin_balloon_bgra = cv2.cvtColor(bin_balloon, cv2.COLOR_GRAY2BGRA)
            bin_balloon_bgra[:, :, 3] = balloon_img[:, :, 3]
            
            # エッジマーキング
            marked_img = bin_balloon_bgra.copy()
            
            for y in range(h):
                for x in range(w):
                    p = marked_img[y, x]
                    
                    # 端から一定距離内または透明画素の周辺をマーク
                    if (x <= balloon_px_th or y <= balloon_px_th or 
                        x >= w - balloon_px_th or y >= h - balloon_px_th):
                        if p[3] != 0:
                            marked_img[y, x] = [255, 0, 0, p[3]]
                    else:
                        # 周辺透明画素チェック
                        is_edge = False
                        for th in range(1, balloon_px_th + 1):
                            neighbors = []
                            if y - th >= 0:
                                neighbors.append(marked_img[y - th, x])
                            if y + th < h:
                                neighbors.append(marked_img[y + th, x])
                            if x - th >= 0:
                                neighbors.append(marked_img[y, x - th])
                            if x + th < w:
                                neighbors.append(marked_img[y, x + th])
                            
                            for neighbor in neighbors:
                                if neighbor[3] == 0:
                                    is_edge = True
                                    break
                            if is_edge:
                                break
                        
                        if is_edge and p[3] != 0:
                            marked_img[y, x] = [255, 0, 0, p[3]]
            
            # 黒画素カウント
            edge_black_count = 0
            black_count = 0
            
            for y in range(h):
                for x in range(w):
                    p = marked_img[y, x]
                    
                    if p[3] != 0:
                        # 隣接赤マークチェック
                        is_adjacent_to_red = False
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                neighbor = marked_img[ny, nx]
                                if (neighbor[0] == 255 and neighbor[1] == 0 and 
                                    neighbor[2] == 0):
                                    is_adjacent_to_red = True
                                    break
                        
                        # エッジ黒画素カウント
                        if (is_adjacent_to_red and 
                            not (p[0] == 255 and p[1] == 0 and p[2] == 0) and
                            p[0] == 0 and p[1] == 0 and p[2] == 0):
                            edge_black_count += 1
                        
                        # 全体黒画素カウント
                        if p[0] == 0 and p[1] == 0 and p[2] == 0:
                            black_count += 1
            
            # 判定
            if edge_black_count == 0 and black_count >= 100:
                filtered_balloons.append(balloon)
        
        return filtered_balloons
    
    def process_images(self):
        """メイン処理"""
        image_paths = self.get_image_paths()
        
        all_panels = []
        all_balloons = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing: {image_path}")
            
            # 画像読み込み
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # ページ分割
            pages = self.page_cut(img)
            
            for j, page in enumerate(pages):
                # ページ分類
                is_black_page = self.get_page_type(page)
                if is_black_page:
                    print(f"  Skipping black page {i}_{j}")
                    continue
                
                print(f"  Processing page {i}_{j}")
                
                # フレーム検出
                panels = self.frame_detect(page)
                
                for k, panel in enumerate(panels):
                    panel.page_idx = j
                    panel.panel_idx = k
                    all_panels.append(panel)
                    
                    # パネル保存
                    panel_filename = f"{i:03d}_{j}_{k}.png"
                    cv2.imwrite(str(self.panels_dir / panel_filename), panel.image)
                    
                    # 吹き出し検出
                    balloons = self.speechballoon_detect(panel.image)
                    
                    if len(balloons) > self.speechballoon_max:
                        continue
                    
                    # 誤検出除去
                    filtered_balloons = self.remove_false_balloons(balloons)
                    
                    for l, balloon in enumerate(filtered_balloons):
                        balloon.panel_idx = len(all_panels) - 1
                        all_balloons.append(balloon)
                        
                        # 吹き出し保存
                        balloon_filename = f"{i:03d}_{j}_{k}_{l}.png"
                        cv2.imwrite(str(self.balloons_dir / balloon_filename), balloon.image)
        
        print(f"Processing complete!")
        print(f"Total panels: {len(all_panels)}")
        print(f"Total balloons: {len(all_balloons)}")
        
        return all_panels, all_balloons


def main():
    if len(sys.argv) != 3:
        print("Usage: python manga_processor.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    processor = MangaProcessor(input_folder, output_folder)
    panels, balloons = processor.process_images()


if __name__ == "__main__":
    main()