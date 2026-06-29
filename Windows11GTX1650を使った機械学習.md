# Windows 11 上で GeForce GTX 1650 Ti を使った機械学習を動かす

## 概要

本書は、Windows 11 環境において NVIDIA GeForce GTX 1650 Ti GPU を使用し、
Keras + TensorFlow による機械学習（文字レベル LSTM 言語モデル）を GPU で動かすための
セットアップ手順をまとめたものです。

---

## 環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce GTX 1650 Ti（VRAM 4GB） |
| RAM | 32GB |
| Python | 3.10.11 |
| TensorFlow | 2.10.1 |
| Keras | 2.10.0 |
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1 |

### バージョンの組み合わせについて

TensorFlow 2.11 以降、Windows Native では GPU サポートが終了しているため、
**TensorFlow 2.10.x が Windows で GPU を使える最終バージョン**となる。
そのため CUDA・cuDNN のバージョンも下記の組み合わせに固定する。

| ソフトウェア | バージョン |
|---|---|
| TensorFlow | 2.10.1 |
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1 |
| Python | 3.9 または 3.10 |

---

## セットアップ手順

### Step 1：NVIDIA ドライバの確認

コマンドプロンプトで以下を実行し、ドライバと CUDA のバージョンを確認する。

```cmd
nvidia-smi
```

`CUDA Version` の欄に表示される数値は、ドライバが対応できる **最大** の CUDA バージョンであり、
実際にインストールする CUDA Toolkit のバージョンとは異なっていても問題ない。

（本環境での確認結果）
```
Driver Version: 512.89       CUDA Version: 11.6
```

最新ドライバが必要な場合は [NVIDIA ドライバダウンロードページ](https://www.nvidia.com/drivers) から入手する。

---

### Step 2：CUDA Toolkit 11.2.2 のインストール

[NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) から
**CUDA Toolkit 11.2.2**（2021年3月リリース）をダウンロードしてインストールする。

同じ 11.2.x 系の中では最もバグ修正が多い 11.2.2 を選ぶこと。

---

### Step 3：cuDNN 8.1 のインストール

[NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) から
**cuDNN 8.1 for CUDA 11.2** をダウンロードする（NVIDIA アカウントが必要）。

ダウンロードしたアーカイブを展開し、中のフォルダを CUDA のインストール先にコピーする。

| コピー元 | コピー先 |
|---|---|
| `bin\` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\` |
| `include\` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\` |
| `lib\` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\` |

---

### Step 4：環境変数（PATH）の設定

Windowsの検索で「環境変数」→「システム環境変数の編集」→「環境変数」を開き、
**システム環境変数** の `Path` に以下の 2 つを追加する。

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
```

設定後はコマンドプロンプトを一度閉じて開き直す。

---

### Step 5：Python 仮想環境の作成とパッケージのインストール

```cmd
python -m venv tf_gpu_env
tf_gpu_env\Scripts\activate
pip install tensorflow==2.10.1
```

TensorFlow 2.10.x には Keras 2.10 が同梱されているため、Keras を別途インストールする必要はない。

#### NumPy のバージョンに注意

TensorFlow 2.10 は NumPy 1.x にのみ対応している。
`pip install tensorflow==2.10.1` を実行した後、NumPy が 2.x 系になっている場合は
以下でダウングレードする。

```cmd
pip install "numpy<2"
```

NumPy が 2.x 系のままだと、TensorFlow の import 時に以下のようなエラーが発生する。

```
AttributeError: _ARRAY_API not found
```

---

### Step 6：GPU 認識の確認

仮想環境を有効化した状態で Python を起動し、以下を実行する。

```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
```

以下のように表示されれば GPU の認識に成功している。

```
2.10.1
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Keras を使ったコードの書き方

TensorFlow 2.x では Keras が同梱されており、import のパスを変更するだけで使える。
スタンドアロン版 Keras（keras 3.x）と混在するとエラーになるため、
**`tensorflow.keras` を明示する形式**を使うこと。

| 旧来の書き方 | 推奨する書き方 |
|---|---|
| `from keras.layers import ...` | `from tensorflow.keras.layers import ...` |
| `from keras.models import ...` | `from tensorflow.keras.models import ...` |
| `from keras.preprocessing.sequence import ...` | `from tensorflow.keras.preprocessing.sequence import ...` |

API・使い勝手は同一であり、モデル定義やコンパイル・学習のコードは変更不要。

---

## GPU を使って学習していることの確認

学習中に別のコマンドプロンプトで以下を実行すると、GPU の使用状況をリアルタイムで確認できる。

```cmd
nvidia-smi
```

GPU が使われている場合は `GPU-Util` が 0% より大きくなり、`Memory-Usage` が増加する。

（本環境での学習中の確認結果）
```
Memory-Usage : 2759MiB / 4096MiB
GPU-Util     : 47%
温度         : 58℃
```

また、TensorFlow のログに以下のように表示されれば cuDNN も正常に動作している。

```
Loaded cuDNN version 8101
```

---

## トラブルシューティング

### TensorFlow の import に失敗する（AttributeError: _ARRAY_API not found）

**原因：** NumPy が 2.x 系になっている。

**対処：**
```cmd
pip install "numpy<2"
```

### GPU が認識されない（list_physical_devices('GPU') が空）

**原因：** CUDA の DLL が PATH に含まれていない。

**確認：**
```cmd
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\cudart64_110.dll"
```

ファイルが存在する場合は Step 4 の PATH 設定を確認する。
ファイルが存在しない場合は CUDA Toolkit の再インストールを行う。

### `cannot import name 'pad_sequences' from 'keras.preprocessing.sequence'`

**原因：** `from keras.xxx` の形式を使っている。

**対処：** 上記「Keras を使ったコードの書き方」の通り、`from tensorflow.keras.xxx` に変更する。

---

## 参考リンク

- [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
- [TensorFlow GPU サポートガイド](https://www.tensorflow.org/install/gpu)
