# Manga Processing System

このプロジェクトは、田中海斗さんのC++コードをPythonで再現し、マンガ画像からコマ（フレーム）と吹き出しを自動検出・抽出するシステムです。

## 概要

C++の以下のファイル群をPythonで統合的に再現：
- `main.cpp` - メイン処理フロー
- `speechballoon_separation.cpp` - 吹き出し検出アルゴリズム  
- `frame_separation.cpp` - フレーム検出アルゴリズム
- その他関連ヘッダーファイル

## 主な機能

1. **ページ分割**: 見開き画像を左右のページに分割
2. **ページ分類**: 白ページ/黒ページの自動判定
3. **フレーム検出**: コマ（フレーム）の検出と抽出
4. **吹き出し検出**: 吹き出しの検出、分類、誤検出除去

## ファイル構成

```
speechBalloon_detector/
├── manga_processor.py          # メインの統合処理システム
├── test_manga_processor.py     # テスト用スクリプト
├── balloon_detect.py          # 単体の吹き出し検出（開発用）
├── test.py                    # 初期テストファイル
├── speechballoon_separation.cpp # 元のC++コード
├── frame_separation.cpp        # 元のC++コード
├── main.cpp                   # 元のC++コード
└── README.md                  # このファイル
```

## 使用方法

### 1. 基本的な使用方法

```bash
# フォルダ内の全画像を処理
python manga_processor.py <input_folder> <output_folder>

# 例：
python manga_processor.py ../manga_images/ ./results/
```

### 2. 単一画像でのテスト

```bash
# テスト用スクリプトを実行
python test_manga_processor.py

# デバッグモード（処理ステップを可視化）
python test_manga_processor.py debug
```

### 3. 単体での吹き出し検出

```bash
# 吹き出し検出のみを実行
python balloon_detect.py
```

## 出力結果

実行後、指定した出力フォルダに以下が生成されます：

```
output_folder/
├── panels/                    # 検出されたコマ画像
│   ├── 000_0_0.png           # ファイル名: <画像番号>_<ページ>_<コマ番号>
│   ├── 000_0_1.png
│   └── ...
└── balloons/                  # 検出された吹き出し画像
    ├── 000_0_0_0.png         # ファイル名: <画像番号>_<ページ>_<コマ番号>_<吹き出し番号>
    ├── 000_0_0_1.png
    └── ...
```

## アルゴリズムの詳細

### フレーム検出アルゴリズム

1. **前処理**: グレースケール変換、ガウシアンフィルタ
2. **吹き出し除去**: 吹き出し候補を検出して塗りつぶし
3. **エッジ検出**: Cannyエッジ検出
4. **直線検出**: Hough変換による直線検出
5. **領域検出**: 論理積による候補領域抽出
6. **後処理**: バウンディングボックス補正、透明化処理

### 吹き出し検出アルゴリズム

1. **前処理**: 二値化、モルフォロジー処理
2. **輪郭検出**: 連結成分ラベリング
3. **幾何フィルタ**: 面積・円形度による一次フィルタリング
4. **光学フィルタ**: 白黒比率による文字領域判定
5. **形状分類**: 円形・矩形・ギザギザの3分類
6. **誤検出除去**: エッジ解析による高精度フィルタリング

## パラメータ調整

主要なパラメータは各クラスの初期化部分で調整可能：

```python
# 二値化閾値
bin_thresh = 230

# 面積フィルタ
min_area_ratio = 0.01  # 最小面積比率
max_area_ratio = 0.9   # 最大面積比率

# 円形度フィルタ
min_circularity = 0.4

# 白黒比率フィルタ
min_bw_ratio = 0.01
max_bw_ratio = 0.7
```

## 実行例と結果

### テスト実行例

```bash
$ python test_manga_processor.py
Loaded image: ./../manga_109_all/collected_images/000325.jpg
Image shape: (1170, 1654, 3)

=== Page Cut Test ===
Split into 2 pages
Page 0: (1170, 827, 3)
Page 1: (1170, 827, 3)

=== Processing Page 0 ===
Page type: White
Detecting frames...
Detected 3 panels
  Detecting balloons in panel 0...
  Found 2 balloon candidates
  After filtering: 1 balloons
    - Type: Zigzag
    - Area: 32700
    - Circularity: 0.477
    - B/W ratio: 0.017
...

Test completed! Check output in: ./test_output
```

### 検出精度

- **コマ検出**: 複雑な形状のコマも高精度で検出
- **吹き出し検出**: 3つの形状（円形・矩形・ギザギザ）を自動分類
- **誤検出除去**: C++アルゴリズムの高精度フィルタリングを再現

## 依存関係

```
numpy>=1.19.0
opencv-python>=4.5.0
```

## C++からの移植について

### 主な変更点

1. **メモリ管理**: C++のポインタ → Pythonのリスト・NumPy配列
2. **画像処理**: OpenCV C++ API → OpenCV Python API
3. **データ構造**: 構造体 → dataclass
4. **ファイルI/O**: C++ストリーム → Pathlib

### 保持された機能

- 全ての検出アルゴリズムロジック
- パラメータ設定
- 出力画像形式（RGBA with透明化）
- 誤検出除去の高精度フィルタリング

## 今後の拡張可能性

- [ ] GUIインターフェース追加
- [ ] バッチ処理の並列化
- [ ] 機械学習モデルとの統合
- [ ] 検出精度の定量評価機能
- [ ] パラメータ自動調整機能