import cv2
import numpy as np

def judge_area(area, thresh=50):
    return area > thresh

# C++と同じグレースケール読み込み
img = cv2.imread('./../manga_109_all/collected_images/000325.jpg', 0)
#前処理
gray = img  # 既にグレースケールなので変換不要
_, bin_img = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
bin_img = cv2.erode(bin_img, kernel, iterations=1)
bin_img = cv2.dilate(bin_img, kernel, iterations=1)
# cv2.imshow("Binarized Image", bin_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# inv = cv2.bitwise_not(bin_img)
# cv2.floodFill(inv, None, (0, 0), 0)      # 角から塗る
# bin_img = cv2.bitwise_not(inv)
# cv2.imshow("Flood Filled Image", bin_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#輪郭抽出
balloons = []
contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 画像サイズを取得
panel_area = img.shape[0] * img.shape[1]

for cnt in contours:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    
    # 周囲長が0の場合はスキップ
    if peri == 0:
        continue
        
    en = 4.0 * np.pi * area / (peri * peri)  # 円形度
    
    # C++と同じ条件での面積と円形度フィルタリング
    if panel_area * 0.01 <= area < panel_area * 0.9 and en > 0.4:
        # バウンディングボックスの取得
        bounding_box = cv2.boundingRect(cnt)
        x, y, w, h = bounding_box
        
        # 吹き出し候補の中心座標
        center_x = x + w // 2
        center_y = y + h // 2
        
        # マスク画像を作成（C++と同じ処理）
        mask = np.full(gray.shape, 255, dtype=np.uint8)  # 初期値255で初期化
        cv2.drawContours(mask, [cnt], -1, 0, 4)  # 輪郭を太さ4で0（黒）で描画
        cv2.drawContours(mask, [cnt], -1, 0, -1)  # 内部を0（黒）で塗りつぶし
        
        # C++のcopyTo処理を再現：マスクを使って背景をグレー（150）にする
        back_150 = np.full(gray.shape, 150, dtype=np.uint8)
        masked_img = gray.copy()
        # マスクが0（黒）の部分にback_150をコピー（C++のcopyTo処理）
        masked_img = np.where(mask == 0, back_150, gray)
        
        # バウンディングボックスで切り出し
        cropped_region = masked_img[y:y+h, x:x+w]
        
        # 白黒比率の計算
        TH = 255 // 3  # 85
        B = np.sum((cropped_region < TH) & (cropped_region != 150))  # 黒画素
        W = np.sum(cropped_region > (255 - TH))  # 白画素
        G = np.sum(cropped_region == 150)  # 背景
        
        # 矩形度の計算
        max_rect = w * h * 0.95
        
        # 吹き出しの白黒比率判定
        if W > 0 and 0.01 < (B / W) < 0.7 and B >= 10:
            # 吹き出しの形状判定
            set_type = 0  # デフォルトは円形
            if area >= max_rect:
                set_type = 1  # 矩形
                print("矩形型吹き出し検出")
            elif en >= 0.7:
                set_type = 0  # 円形
                print("円形型吹き出し検出")
            else:
                set_type = 2  # ギザギザ
                print("ギザギザ型吹き出し検出")
            
            # コマサイズに近いものを除去
            if w * h < (img.shape[0] * img.shape[1]) * 0.9:
                # 元画像から吹き出し部分を切り出し
                balloon_roi = img[y:y+h, x:x+w].copy()
                
                # マスクも切り出し
                mask_roi = mask[y:y+h, x:x+w]
                
                # 背景を透明にした画像を作成（RGBAに変換）
                # グレースケールからBGRAに変換
                balloon_rgba = cv2.cvtColor(balloon_roi, cv2.COLOR_GRAY2BGRA)
                balloon_rgba[:, :, 3] = mask_roi  # アルファチャンネルにマスクを設定
                
                balloons.append({
                    'image': balloon_rgba,
                    'bbox': bounding_box,
                    'contour': cnt,  # 輪郭情報を追加
                    'center': (center_x, center_y),
                    'area': area,
                    'circularity': en,
                    'type': set_type,
                    'bw_ratio': B / W if W > 0 else 0
                })
                
                print(f"吹き出し検出: 面積={area:.0f}, 円形度={en:.3f}, 白黒比={B/W:.3f}")

print(f"検出された吹き出し数: {len(balloons)}")

# 分類ごとの統計を表示
if len(balloons) > 0:
    type_counts = {0: 0, 1: 0, 2: 0}  # 円形, 矩形, ギザギザ
    type_names_stats = {0: "円形", 1: "矩形", 2: "ギザギザ"}
    
    for balloon in balloons:
        balloon_type = balloon['type']
        if balloon_type in type_counts:
            type_counts[balloon_type] += 1
    
    print("--- 分類別統計 ---")
    for type_id, count in type_counts.items():
        if count > 0:
            print(f"{type_names_stats[type_id]}: {count}個")
    print("------------------")

# 結果の可視化
if len(balloons) > 0:
    # 分類ごとの色を定義
    type_colors = {
        0: (0, 255, 255),    # 円形: 黄色 (BGR)
        1: (255, 0, 0),      # 矩形: 青色 (BGR)
        2: (0, 255, 0)       # ギザギザ: 緑色 (BGR)
    }
    type_names = {
        0: "Circle",
        1: "Rect",
        2: "Zigzag"
    }
    
    # 元画像に輪郭を描画（グレースケールからBGRに変換）
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, balloon in enumerate(balloons):
        x, y, w, h = balloon['bbox']
        balloon_type = balloon['type']
        color = type_colors.get(balloon_type, (128, 128, 128))  # デフォルト色
        type_name = type_names.get(balloon_type, "Unknown")
        
        # 保存された輪郭情報を使用
        cv2.drawContours(result_img, [balloon['contour']], -1, color, 3)
        cv2.putText(result_img, f'{i}:{type_name}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 凡例を追加
    legend_y_start = 30
    cv2.putText(result_img, "Legend:", (10, legend_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_img, "Circle", (10, legend_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors[0], 2)
    cv2.putText(result_img, "Rect", (10, legend_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors[1], 2)
    cv2.putText(result_img, "Zigzag", (10, legend_y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors[2], 2)
    
    # リサイズして表示
    height, width = result_img.shape[:2]
    if height > 800:
        scale = 800 / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        result_img = cv2.resize(result_img, (new_width, new_height))
    
    cv2.imshow("Detected Speech Balloons", result_img)
    cv2.imwrite("detected_balloons.png", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 各吹き出しを個別に保存
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, balloon in enumerate(balloons):
        filename = f"{output_dir}/balloon_{i:02d}.png"
        cv2.imwrite(filename, balloon['image'])
        print(f"保存完了: {filename}")
else:
    print("吹き出しが検出されませんでした。")

# 誤検出除去処理（C++の後半部分を参考）
def remove_false_balloons(balloons):
    """C++コードの誤検出除去ロジックを実装"""
    balloon_px_th = 5  # 画像端から何ピクセル内側まで判断するか
    filtered_balloons = []
    
    for balloon in balloons:
        balloon_img = balloon['image']
        h, w = balloon_img.shape[:2]
        
        # BGRAをグレースケールに変換
        gray_balloon = cv2.cvtColor(balloon_img, cv2.COLOR_BGRA2GRAY)
        
        # 二値化
        _, bin_balloon = cv2.threshold(gray_balloon, 150, 255, cv2.THRESH_BINARY)
        bin_balloon_bgra = cv2.cvtColor(bin_balloon, cv2.COLOR_GRAY2BGRA)
        
        # アルファチャンネルを元の画像からコピー
        bin_balloon_bgra[:, :, 3] = balloon_img[:, :, 3]
        
        # 画像端からの距離に基づく領域をマーク
        marked_img = bin_balloon_bgra.copy()
        
        for y in range(h):
            for x in range(w):
                p = marked_img[y, x]
                
                # 画像端からballoon_px_th以内、または透明でない画素の周辺をチェック
                if (x <= balloon_px_th or y <= balloon_px_th or 
                    x >= w - balloon_px_th or y >= h - balloon_px_th):
                    if p[3] != 0:  # 透明でない場合
                        marked_img[y, x] = [255, 0, 0, p[3]]  # 赤でマーク
                else:
                    # 周辺画素をチェック
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
                        
                        # 対角線方向もチェック
                        if y - th >= 0 and x + th < w:
                            neighbors.append(marked_img[y - th, x + th])
                        if y - th >= 0 and x - th >= 0:
                            neighbors.append(marked_img[y - th, x - th])
                        if y + th < h and x + th < w:
                            neighbors.append(marked_img[y + th, x + th])
                        if y + th < h and x - th >= 0:
                            neighbors.append(marked_img[y + th, x - th])
                        
                        for neighbor in neighbors:
                            if neighbor[3] == 0:  # 透明画素があれば
                                is_edge = True
                                break
                        if is_edge:
                            break
                    
                    if is_edge and p[3] != 0:
                        marked_img[y, x] = [255, 0, 0, p[3]]  # 赤でマーク
        
        # 赤マークと接している黒画素をチェック
        edge_black_count = 0
        black_count = 0
        white_count = 0
        
        for y in range(h):
            for x in range(w):
                p = marked_img[y, x]
                
                if p[3] != 0:  # 透明でない画素
                    # 隣接画素をチェック
                    is_adjacent_to_red = False
                    if y > 0:
                        neighbor = marked_img[y - 1, x]
                        if neighbor[0] == 255 and neighbor[1] == 0 and neighbor[2] == 0:
                            is_adjacent_to_red = True
                    if y < h - 1:
                        neighbor = marked_img[y + 1, x]
                        if neighbor[0] == 255 and neighbor[1] == 0 and neighbor[2] == 0:
                            is_adjacent_to_red = True
                    if x > 0:
                        neighbor = marked_img[y, x - 1]
                        if neighbor[0] == 255 and neighbor[1] == 0 and neighbor[2] == 0:
                            is_adjacent_to_red = True
                    if x < w - 1:
                        neighbor = marked_img[y, x + 1]
                        if neighbor[0] == 255 and neighbor[1] == 0 and neighbor[2] == 0:
                            is_adjacent_to_red = True
                    
                    # 赤マークと隣接する黒画素をカウント
                    if (is_adjacent_to_red and 
                        not (p[0] == 255 and p[1] == 0 and p[2] == 0) and
                        p[0] == 0 and p[1] == 0 and p[2] == 0):
                        edge_black_count += 1
                    
                    # 全体の黒白画素をカウント
                    if p[0] == 0 and p[1] == 0 and p[2] == 0:
                        black_count += 1
                    else:
                        white_count += 1
        
        # 誤検出判定
        if edge_black_count == 0 and black_count >= 100:
            filtered_balloons.append(balloon)
            print(f"吹き出し保持: 黒画素={black_count}, エッジ黒画素={edge_black_count}")
        else:
            print(f"吹き出し除去: 黒画素={black_count}, エッジ黒画素={edge_black_count}")
    
    return filtered_balloons

# # 誤検出除去を適用
# print("\n--- 誤検出除去処理 ---")
# filtered_balloons = remove_false_balloons(balloons)
# print(f"フィルタリング後の吹き出し数: {len(filtered_balloons)}")

# # フィルタリング後の分類統計
# if len(filtered_balloons) > 0:
#     filtered_type_counts = {0: 0, 1: 0, 2: 0}
#     type_names_stats = {0: "円形", 1: "矩形", 2: "ギザギザ"}
    
#     for balloon in filtered_balloons:
#         balloon_type = balloon['type']
#         if balloon_type in filtered_type_counts:
#             filtered_type_counts[balloon_type] += 1
    
#     print("--- フィルタリング後分類別統計 ---")
#     for type_id, count in filtered_type_counts.items():
#         if count > 0:
#             print(f"{type_names_stats[type_id]}: {count}個")
#     print("-----------------------------")

# # フィルタリング後の結果を保存
# if len(filtered_balloons) > 0:
#     import os
#     filtered_output_dir = "output/filtered"
#     if not os.path.exists(filtered_output_dir):
#         os.makedirs(filtered_output_dir)
    
#     for i, balloon in enumerate(filtered_balloons):
#         filename = f"{filtered_output_dir}/balloon_{i:02d}.png"
#         cv2.imwrite(filename, balloon['image'])
#         print(f"フィルタリング済み保存: {filename}")
    
#     # フィルタリング後の結果を可視化
#     result_filtered_img = img.copy()
    
#     # 分類ごとの色を定義（フィルタリング後用）
#     type_colors_filtered = {
#         0: (0, 255, 255),    # 円形: 黄色 (BGR)
#         1: (255, 0, 0),      # 矩形: 青色 (BGR)
#         2: (0, 255, 0)       # ギザギザ: 緑色 (BGR)
#     }
#     type_names_filtered = {
#         0: "Circle",
#         1: "Rect", 
#         2: "Zigzag"
#     }
    
#     for i, balloon in enumerate(filtered_balloons):
#         x, y, w, h = balloon['bbox']
#         balloon_type = balloon['type']
#         color = type_colors_filtered.get(balloon_type, (0, 0, 255))  # デフォルト赤色
#         type_name = type_names_filtered.get(balloon_type, "Unknown")
        
#         cv2.rectangle(result_filtered_img, (x, y), (x + w, y + h), color, 3)
#         cv2.putText(result_filtered_img, f'F{i}:{type_name}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
#     # フィルタリング後の凡例を追加
#     legend_y_start = 30
#     cv2.putText(result_filtered_img, "Filtered Legend:", (10, legend_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#     cv2.putText(result_filtered_img, "Circle", (10, legend_y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors_filtered[0], 2)
#     cv2.putText(result_filtered_img, "Rect", (10, legend_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors_filtered[1], 2)
#     cv2.putText(result_filtered_img, "Zigzag", (10, legend_y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_colors_filtered[2], 2)
    
#     # リサイズして表示
#     height, width = result_filtered_img.shape[:2]
#     if height > 800:
#         scale = 800 / height
#         new_width = int(width * scale)
#         new_height = int(height * scale)
#         result_filtered_img = cv2.resize(result_filtered_img, (new_width, new_height))
    
#     cv2.imshow("Filtered Speech Balloons", result_filtered_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
