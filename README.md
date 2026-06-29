I am trying out machine learning using Keras. 
Keras has an intuitive API, making it relatively easy to learn.
However, since I will be using a GTX 1650 GPU on Windows 11, I have to use a slightly older version of Keras.

# Addition Learning Using LSTM (Verification of Generalization)
```
(tf_gpu_env) C:\Users\user\Keras>py lstm_add_seq2seq7.py
モデル構築中...
2026-06-29 09:18:04.979807: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-06-29 09:18:05.436363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2151 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1650 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5
Model: "seq2seq_train"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 encoder_input (InputLayer)     [(None, None, 12)]   0           []

 decoder_input (InputLayer)     [(None, None, 12)]   0           []

 encoder_lstm (LSTM)            [(None, 512),        1075200     ['encoder_input[0][0]']
                                 (None, 512),
                                 (None, 512)]

 decoder_lstm (LSTM)            [(None, None, 512),  1075200     ['decoder_input[0][0]',
                                 (None, 512),                     'encoder_lstm[0][1]',
                                 (None, 512)]                     'encoder_lstm[0][2]']

 decoder_dense (Dense)          (None, None, 12)     6156        ['decoder_lstm[0][0]']

==================================================================================================
Total params: 2,156,556
Trainable params: 2,156,556
Non-trainable params: 0
__________________________________________________________________________________________________

============================================================
カリキュラム学習開始（汎化確認強化版）
  MIN_EPOCHS=20, MAX_EPOCHS=[30, 200, 200, 200, 200, 200], PATIENCE=20
  HOLDOUT_RATIO=0.2 → 各ステージの20%を未学習テスト用に確保
============================================================

--- Stage 1: 1桁  全55通り（全件訓練 + 20%ホールドアウト）---
  訓練:6400件  バリデーション:11件  ホールドアウト:11件
Epoch 1/30
2026-06-29 09:18:15.718785: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101
50/50 [==============================] - 28s 40ms/step - loss: 0.6227 - accuracy: 0.8465 - val_loss: 0.3892 - val_accuracy: 0.8750
Epoch 2/30
50/50 [==============================] - 1s 24ms/step - loss: 0.2892 - accuracy: 0.8921 - val_loss: 0.3230 - val_accuracy: 0.8977
Epoch 3/30
50/50 [==============================] - 1s 24ms/step - loss: 0.1592 - accuracy: 0.9465 - val_loss: 0.3846 - val_accuracy: 0.9205
Epoch 4/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0738 - accuracy: 0.9767 - val_loss: 0.4655 - val_accuracy: 0.9205
Epoch 5/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0941 - accuracy: 0.9689 - val_loss: 0.3318 - val_accuracy: 0.9432
Epoch 6/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0310 - accuracy: 0.9920 - val_loss: 0.3509 - val_accuracy: 0.9545
Epoch 7/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0179 - accuracy: 0.9949 - val_loss: 0.3631 - val_accuracy: 0.9545
Epoch 8/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0117 - accuracy: 0.9976 - val_loss: 0.3566 - val_accuracy: 0.9659
Epoch 9/30
50/50 [==============================] - 1s 24ms/step - loss: 0.0680 - accuracy: 0.9854 - val_loss: 0.5310 - val_accuracy: 0.9432
Epoch 10/30
50/50 [==============================] - 1s 25ms/step - loss: 0.0377 - accuracy: 0.9889 - val_loss: 0.3085 - val_accuracy: 0.9659

...

50/50 [==============================] - 1s 24ms/step - loss: 4.7827e-04 - accuracy: 1.0000 - val_loss: 0.3741 - val_accuracy: 0.9432
Epoch 21/30
50/50 [==============================] - 1s 24ms/step - loss: 4.1628e-04 - accuracy: 1.0000 - val_loss: 0.3779 - val_accuracy: 0.9432
Epoch 22/30
50/50 [==============================] - 1s 24ms/step - loss: 3.6284e-04 - accuracy: 1.0000 - val_loss: 0.3806 - val_accuracy: 0.9432
Epoch 23/30
50/50 [==============================] - 1s 24ms/step - loss: 3.2033e-04 - accuracy: 1.0000 - val_loss: 0.3848 - val_accuracy: 0.9432
Epoch 24/30
50/50 [==============================] - 1s 24ms/step - loss: 2.8388e-04 - accuracy: 1.0000 - val_loss: 0.3857 - val_accuracy: 0.9432
Epoch 25/30
50/50 [==============================] - 1s 24ms/step - loss: 2.5353e-04 - accuracy: 1.0000 - val_loss: 0.3883 - val_accuracy: 0.9432
Epoch 26/30
50/50 [==============================] - 1s 24ms/step - loss: 2.2762e-04 - accuracy: 1.0000 - val_loss: 0.3925 - val_accuracy: 0.9432
Epoch 27/30
50/50 [==============================] - 1s 24ms/step - loss: 2.0551e-04 - accuracy: 1.0000 - val_loss: 0.3950 - val_accuracy: 0.9432
Epoch 28/30
50/50 [==============================] - 1s 24ms/step - loss: 1.8642e-04 - accuracy: 1.0000 - val_loss: 0.3959 - val_accuracy: 0.9432
Epoch 29/30
50/50 [==============================] - 1s 25ms/step - loss: 1.6995e-04 - accuracy: 1.0000 - val_loss: 0.3973 - val_accuracy: 0.9432
Epoch 30/30
50/50 [==============================] - 1s 24ms/step - loss: 1.5557e-04 - accuracy: 1.0000 - val_loss: 0.4013 - val_accuracy: 0.9432
  → ステージ1 完了: 30エポック実行

  ◆ [Stage 1 (1桁)] 汎化テスト（未学習11件）: 6/11 = 54.5%
  　正解例（最大5件）:
    Q:            1+7  正解: 8        予測: 8        〇
    Q:            4+6  正解: 10       予測: 10       〇
    Q:            1+9  正解: 10       予測: 10       〇
    Q:            3+5  正解: 8        予測: 8        〇
    Q:            7+8  正解: 15       予測: 15       〇
  　不正解例（最大5件）:
    Q:            0+3  正解: 3        予測: 2        ×
    Q:            7+9  正解: 16       予測: 17       ×
    Q:            0+2  正解: 2        予測: 0        ×
    Q:            0+1  正解: 1        予測: 0        ×
    Q:            1+2  正解: 3        予測: 2        ×
  ★ 全55通り正解率: 50/55=90.9%

--- Stage 2: 2桁  サンプル数=5050（うち20%ホールドアウト）---
  訓練:3712件  バリデーション:505件  ホールドアウト:505件
Epoch 1/200
50/50 [==============================] - 2s 28ms/step - loss: 0.9409 - accuracy: 0.7756 - val_loss: 0.6747 - val_accuracy: 0.7715
Epoch 2/200
50/50 [==============================] - 1s 26ms/step - loss: 0.5663 - accuracy: 0.8039 - val_loss: 0.6163 - val_accuracy: 0.7780
Epoch 3/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4938 - accuracy: 0.8202 - val_loss: 0.5702 - val_accuracy: 0.7879
Epoch 4/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4714 - accuracy: 0.8253 - val_loss: 0.5780 - val_accuracy: 0.7923
Epoch 5/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4649 - accuracy: 0.8267 - val_loss: 0.5662 - val_accuracy: 0.7983
Epoch 6/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4398 - accuracy: 0.8344 - val_loss: 0.5240 - val_accuracy: 0.7960
Epoch 7/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4258 - accuracy: 0.8369 - val_loss: 0.5623 - val_accuracy: 0.7886
Epoch 8/200
50/50 [==============================] - 1s 26ms/step - loss: 0.4078 - accuracy: 0.8422 - val_loss: 0.5067 - val_accuracy: 0.8047
Epoch 9/200
50/50 [==============================] - 1s 26ms/step - loss: 0.3726 - accuracy: 0.8530 - val_loss: 0.4725 - val_accuracy: 0.8223
Epoch 10/200
50/50 [==============================] - 1s 26ms/step - loss: 0.3279 - accuracy: 0.8707 - val_loss: 0.4478 - val_accuracy: 0.8329
Epoch 11/200
50/50 [==============================] - 1s 26ms/step - loss: 0.3004 - accuracy: 0.8755 - val_loss: 0.4452 - val_accuracy: 0.8302

...

50/50 [==============================] - 1s 27ms/step - loss: 6.3663e-06 - accuracy: 1.0000 - val_loss: 0.0270 - val_accuracy: 0.9948
Epoch 191/200
50/50 [==============================] - 1s 27ms/step - loss: 6.3291e-06 - accuracy: 1.0000 - val_loss: 0.0271 - val_accuracy: 0.9948
Epoch 192/200
50/50 [==============================] - 1s 26ms/step - loss: 5.8999e-06 - accuracy: 1.0000 - val_loss: 0.0271 - val_accuracy: 0.9948
Epoch 193/200
50/50 [==============================] - 1s 27ms/step - loss: 5.9598e-06 - accuracy: 1.0000 - val_loss: 0.0272 - val_accuracy: 0.9948
Epoch 194/200
50/50 [==============================] - 1s 26ms/step - loss: 5.7515e-06 - accuracy: 1.0000 - val_loss: 0.0270 - val_accuracy: 0.9948
Epoch 195/200
50/50 [==============================] - 1s 27ms/step - loss: 5.5773e-06 - accuracy: 1.0000 - val_loss: 0.0272 - val_accuracy: 0.9948
Epoch 196/200
50/50 [==============================] - 1s 26ms/step - loss: 5.3604e-06 - accuracy: 1.0000 - val_loss: 0.0271 - val_accuracy: 0.9948
Epoch 197/200
50/50 [==============================] - 1s 27ms/step - loss: 5.2620e-06 - accuracy: 1.0000 - val_loss: 0.0273 - val_accuracy: 0.9948
Epoch 198/200
50/50 [==============================] - 1s 26ms/step - loss: 4.9844e-06 - accuracy: 1.0000 - val_loss: 0.0271 - val_accuracy: 0.9950
Epoch 199/200
50/50 [==============================] - 1s 27ms/step - loss: 5.0420e-06 - accuracy: 1.0000 - val_loss: 0.0273 - val_accuracy: 0.9948
Epoch 200/200
50/50 [==============================] - 1s 27ms/step - loss: 4.7920e-06 - accuracy: 1.0000 - val_loss: 0.0275 - val_accuracy: 0.9948
  → ステージ2 完了: 200エポック実行

  ◆ [Stage 2 (2桁)] 汎化テスト（未学習500件）: 479/500 = 95.8%
  　正解例（最大5件）:
    Q:           30+7  正解: 37       予測: 37       〇
    Q:          73+18  正解: 91       予測: 91       〇
    Q:          13+90  正解: 103      予測: 103      〇
    Q:           5+38  正解: 43       予測: 43       〇
    Q:           58+0  正解: 58       予測: 58       〇
  　不正解例（最大5件）:
    Q:           41+9  正解: 50       予測: 59       ×
    Q:           23+9  正解: 32       予測: 31       ×
    Q:           93+8  正解: 101      予測: 91       ×
    Q:           1+14  正解: 15       予測: 16       ×
    Q:           29+9  正解: 38       予測: 48       ×

--- Stage 3: 3桁  サンプル数=20000（うち20%ホールドアウト）---
  訓練:16000件  バリデーション:2000件  ホールドアウト:2000件
Epoch 1/200
50/50 [==============================] - 2s 34ms/step - loss: 1.1869 - accuracy: 0.6935 - val_loss: 0.9065 - val_accuracy: 0.6866
Epoch 2/200
50/50 [==============================] - 2s 32ms/step - loss: 0.8146 - accuracy: 0.7140 - val_loss: 0.8520 - val_accuracy: 0.6957
Epoch 3/200
50/50 [==============================] - 2s 32ms/step - loss: 0.7663 - accuracy: 0.7214 - val_loss: 0.8023 - val_accuracy: 0.7015
Epoch 4/200
50/50 [==============================] - 2s 32ms/step - loss: 0.7234 - accuracy: 0.7291 - val_loss: 0.7738 - val_accuracy: 0.7154
Epoch 5/200
50/50 [==============================] - 2s 32ms/step - loss: 0.6930 - accuracy: 0.7362 - val_loss: 0.7364 - val_accuracy: 0.7190
Epoch 6/200
50/50 [==============================] - 2s 32ms/step - loss: 0.6444 - accuracy: 0.7502 - val_loss: 0.6992 - val_accuracy: 0.7327
Epoch 7/200
50/50 [==============================] - 2s 32ms/step - loss: 0.6328 - accuracy: 0.7523 - val_loss: 0.7003 - val_accuracy: 0.7283
Epoch 8/200
50/50 [==============================] - 2s 32ms/step - loss: 0.5916 - accuracy: 0.7689 - val_loss: 0.6375 - val_accuracy: 0.7552
Epoch 9/200
50/50 [==============================] - 2s 31ms/step - loss: 0.5596 - accuracy: 0.7828 - val_loss: 0.6822 - val_accuracy: 0.7293
Epoch 10/200
50/50 [==============================] - 2s 32ms/step - loss: 0.5652 - accuracy: 0.7786 - val_loss: 0.6431 - val_accuracy: 0.7531

...

50/50 [==============================] - 2s 32ms/step - loss: 1.5649e-04 - accuracy: 1.0000 - val_loss: 5.8571e-04 - val_accuracy: 0.9999
Epoch 191/200
50/50 [==============================] - 2s 32ms/step - loss: 1.5458e-04 - accuracy: 1.0000 - val_loss: 5.9299e-04 - val_accuracy: 0.9999
Epoch 192/200
50/50 [==============================] - 2s 32ms/step - loss: 1.4355e-04 - accuracy: 1.0000 - val_loss: 5.5802e-04 - val_accuracy: 0.9999
Epoch 193/200
50/50 [==============================] - 2s 32ms/step - loss: 1.4055e-04 - accuracy: 1.0000 - val_loss: 5.2233e-04 - val_accuracy: 0.9999
Epoch 194/200
50/50 [==============================] - 2s 32ms/step - loss: 1.3622e-04 - accuracy: 1.0000 - val_loss: 5.2080e-04 - val_accuracy: 0.9999
Epoch 195/200
50/50 [==============================] - 2s 32ms/step - loss: 1.3455e-04 - accuracy: 1.0000 - val_loss: 5.2051e-04 - val_accuracy: 0.9999
Epoch 196/200
50/50 [==============================] - 2s 32ms/step - loss: 1.2366e-04 - accuracy: 1.0000 - val_loss: 5.0604e-04 - val_accuracy: 0.9999
Epoch 197/200
50/50 [==============================] - 2s 32ms/step - loss: 1.2555e-04 - accuracy: 1.0000 - val_loss: 4.9661e-04 - val_accuracy: 0.9999
Epoch 198/200
50/50 [==============================] - 2s 32ms/step - loss: 1.2084e-04 - accuracy: 1.0000 - val_loss: 4.8349e-04 - val_accuracy: 0.9999
Epoch 199/200
50/50 [==============================] - 2s 32ms/step - loss: 1.1569e-04 - accuracy: 1.0000 - val_loss: 4.4851e-04 - val_accuracy: 0.9999
Epoch 200/200
50/50 [==============================] - 2s 31ms/step - loss: 1.1157e-04 - accuracy: 1.0000 - val_loss: 4.8300e-04 - val_accuracy: 0.9999
  → ステージ3 完了: 200エポック実行

  ◆ [Stage 3 (3桁)] 汎化テスト（未学習500件）: 500/500 = 100.0%
  　正解例（最大5件）:
    Q:         605+39  正解: 644      予測: 644      〇
    Q:         357+29  正解: 386      予測: 386      〇
    Q:         21+749  正解: 770      予測: 770      〇
    Q:         844+54  正解: 898      予測: 898      〇
    Q:        828+158  正解: 986      予測: 986      〇

--- Stage 4: 4桁  サンプル数=20000（うち20%ホールドアウト）---
  訓練:16000件  バリデーション:2000件  ホールドアウト:2000件
Epoch 1/200
50/50 [==============================] - 2s 35ms/step - loss: 1.4907 - accuracy: 0.6716 - val_loss: 0.8613 - val_accuracy: 0.6850
Epoch 2/200
50/50 [==============================] - 2s 32ms/step - loss: 0.7250 - accuracy: 0.7318 - val_loss: 0.6634 - val_accuracy: 0.7509
Epoch 3/200
50/50 [==============================] - 2s 32ms/step - loss: 0.5657 - accuracy: 0.7925 - val_loss: 0.5255 - val_accuracy: 0.8119
Epoch 4/200
50/50 [==============================] - 2s 32ms/step - loss: 0.4508 - accuracy: 0.8446 - val_loss: 0.4309 - val_accuracy: 0.8529
Epoch 5/200
50/50 [==============================] - 2s 32ms/step - loss: 0.3725 - accuracy: 0.8739 - val_loss: 0.3703 - val_accuracy: 0.8733
Epoch 6/200
50/50 [==============================] - 2s 32ms/step - loss: 0.3234 - accuracy: 0.8904 - val_loss: 0.3187 - val_accuracy: 0.8934
Epoch 7/200
50/50 [==============================] - 2s 32ms/step - loss: 0.2854 - accuracy: 0.9023 - val_loss: 0.2796 - val_accuracy: 0.9054
Epoch 8/200
50/50 [==============================] - 2s 32ms/step - loss: 0.2416 - accuracy: 0.9197 - val_loss: 0.2365 - val_accuracy: 0.9214
Epoch 9/200
50/50 [==============================] - 2s 32ms/step - loss: 0.2034 - accuracy: 0.9344 - val_loss: 0.2019 - val_accuracy: 0.9359
Epoch 10/200
50/50 [==============================] - 2s 33ms/step - loss: 0.1792 - accuracy: 0.9426 - val_loss: 0.1966 - val_accuracy: 0.9290

...

50/50 [==============================] - 2s 32ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0034 - val_accuracy: 0.9993
Epoch 191/200
50/50 [==============================] - 2s 32ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 0.9993
Epoch 192/200
50/50 [==============================] - 2s 32ms/step - loss: 9.9314e-04 - accuracy: 1.0000 - val_loss: 0.0032 - val_accuracy: 0.9992
Epoch 193/200
50/50 [==============================] - 2s 32ms/step - loss: 9.0937e-04 - accuracy: 1.0000 - val_loss: 0.0031 - val_accuracy: 0.9992
Epoch 194/200
50/50 [==============================] - 2s 33ms/step - loss: 8.7307e-04 - accuracy: 1.0000 - val_loss: 0.0030 - val_accuracy: 0.9994
Epoch 195/200
50/50 [==============================] - 2s 33ms/step - loss: 8.5008e-04 - accuracy: 1.0000 - val_loss: 0.0029 - val_accuracy: 0.9993
Epoch 196/200
50/50 [==============================] - 2s 32ms/step - loss: 8.0250e-04 - accuracy: 1.0000 - val_loss: 0.0029 - val_accuracy: 0.9993
Epoch 197/200
50/50 [==============================] - 2s 32ms/step - loss: 7.7077e-04 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 0.9994
Epoch 198/200
50/50 [==============================] - 2s 32ms/step - loss: 7.4031e-04 - accuracy: 1.0000 - val_loss: 0.0027 - val_accuracy: 0.9994
Epoch 199/200
50/50 [==============================] - 2s 32ms/step - loss: 6.9630e-04 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 0.9994
Epoch 200/200
50/50 [==============================] - 2s 33ms/step - loss: 6.7079e-04 - accuracy: 1.0000 - val_loss: 0.0028 - val_accuracy: 0.9993
  → ステージ4 完了: 200エポック実行

  ◆ [Stage 4 (4桁)] 汎化テスト（未学習500件）: 498/500 = 99.6%
  　正解例（最大5件）:
    Q:          19+43  正解: 62       予測: 62       〇
    Q:      2473+1810  正解: 4283     予測: 4283     〇
    Q:        5803+19  正解: 5822     予測: 5822     〇
    Q:        146+569  正解: 715      予測: 715      〇
    Q:         38+326  正解: 364      予測: 364      〇
  　不正解例（最大5件）:
    Q:      1986+7983  正解: 9969     予測: 9869     ×
    Q:       8013+904  正解: 8917     予測: 9917     ×

--- Stage 5: 5桁  サンプル数=20000（うち20%ホールドアウト）---
  訓練:16000件  バリデーション:2000件  ホールドアウト:2000件
Epoch 1/200
50/50 [==============================] - 2s 36ms/step - loss: 0.9537 - accuracy: 0.7785 - val_loss: 0.4708 - val_accuracy: 0.8393
Epoch 2/200
50/50 [==============================] - 2s 33ms/step - loss: 0.3351 - accuracy: 0.8858 - val_loss: 0.3197 - val_accuracy: 0.8909
Epoch 3/200
50/50 [==============================] - 2s 33ms/step - loss: 0.2458 - accuracy: 0.9165 - val_loss: 0.2539 - val_accuracy: 0.9135
Epoch 4/200
50/50 [==============================] - 2s 32ms/step - loss: 0.2009 - accuracy: 0.9321 - val_loss: 0.2199 - val_accuracy: 0.9236
Epoch 5/200
50/50 [==============================] - 2s 32ms/step - loss: 0.1821 - accuracy: 0.9364 - val_loss: 0.2021 - val_accuracy: 0.9290
Epoch 6/200
50/50 [==============================] - 2s 32ms/step - loss: 0.1687 - accuracy: 0.9402 - val_loss: 0.1907 - val_accuracy: 0.9314
Epoch 7/200
50/50 [==============================] - 2s 33ms/step - loss: 0.1533 - accuracy: 0.9456 - val_loss: 0.1753 - val_accuracy: 0.9367
Epoch 8/200
50/50 [==============================] - 2s 32ms/step - loss: 0.1797 - accuracy: 0.9353 - val_loss: 0.1989 - val_accuracy: 0.9284
Epoch 9/200
50/50 [==============================] - 2s 32ms/step - loss: 0.1488 - accuracy: 0.9464 - val_loss: 0.1671 - val_accuracy: 0.9389
Epoch 10/200
50/50 [==============================] - 2s 33ms/step - loss: 0.1277 - accuracy: 0.9550 - val_loss: 0.1490 - val_accuracy: 0.9454

...


```
