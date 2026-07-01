I am trying out machine learning using Keras. 
Keras has an intuitive API, making it relatively easy to learn.
However, since I will be using a GTX 1650 GPU on Windows 11, I have to use a slightly older version of Keras.

## Addition Learning Using LSTM (Verification of Generalization)
```
(tf_gpu_env) C:\Users\user\Keras>py lstm_add_seq2seq9.py
モデル構築中...
2026-07-02 02:09:02.168051: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-07-02 02:09:02.591354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2151 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1650 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5
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
カリキュラム学習開始（汎化確認強化版 v4）
  MIN_EPOCHS=20, MAX_EPOCHS=[30, 200, 200, 200, 200, 200], PATIENCE=60
  HOLDOUT_RATIO=0.2 → 各ステージの20%を一時的に未学習テスト用に確保
  ※ ホールドアウトは評価後、次ステージの訓練データに合流します
  ※ 最終ステージ（6桁）のホールドアウトのみ、最後まで未学習のまま最終テストに使用します
  ※ [v4] 各ステージ終了後、val_lossが最良だったエポックの重みに復元してから次に進みます
============================================================

--- Stage 1: 1桁  全100通り（全順序対、全件訓練 + 20%ホールドアウト）---
  訓練:6400件  バリデーション:20件  このステージのホールドアウト:20件
Epoch 1/30
2026-07-02 02:09:06.176617: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8101
50/50 [==============================] - 5s 38ms/step - loss: 0.6225 - accuracy: 0.8494 - val_loss: 0.4002 - val_accuracy: 0.8813
Epoch 2/30
50/50 [==============================] - 1s 26ms/step - loss: 0.2973 - accuracy: 0.8881 - val_loss: 0.2766 - val_accuracy: 0.9062
Epoch 3/30
50/50 [==============================] - 1s 27ms/step - loss: 0.1865 - accuracy: 0.9311 - val_loss: 0.2406 - val_accuracy: 0.8875
Epoch 4/30
50/50 [==============================] - 1s 26ms/step - loss: 0.1200 - accuracy: 0.9574 - val_loss: 0.1480 - val_accuracy: 0.9312
Epoch 5/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0751 - accuracy: 0.9754 - val_loss: 0.1583 - val_accuracy: 0.9312
Epoch 6/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0595 - accuracy: 0.9779 - val_loss: 0.0906 - val_accuracy: 0.9688
Epoch 7/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0273 - accuracy: 0.9936 - val_loss: 0.0508 - val_accuracy: 0.9937
Epoch 8/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0146 - accuracy: 0.9979 - val_loss: 0.0272 - val_accuracy: 0.9937
Epoch 9/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0478 - accuracy: 0.9879 - val_loss: 0.4351 - val_accuracy: 0.8813
Epoch 10/30
50/50 [==============================] - 1s 26ms/step - loss: 0.0725 - accuracy: 0.9744 - val_loss: 0.0428 - val_accuracy: 0.9937

...

Epoch 21/30
50/50 [==============================] - 1s 26ms/step - loss: 4.9818e-04 - accuracy: 1.0000 - val_loss: 0.0017 - val_accuracy: 1.0000
Epoch 22/30
50/50 [==============================] - 1s 27ms/step - loss: 4.2939e-04 - accuracy: 1.0000 - val_loss: 0.0014 - val_accuracy: 1.0000
Epoch 23/30
50/50 [==============================] - 1s 26ms/step - loss: 3.7550e-04 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000
Epoch 24/30
50/50 [==============================] - 1s 27ms/step - loss: 3.3089e-04 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
Epoch 25/30
50/50 [==============================] - 1s 26ms/step - loss: 2.9362e-04 - accuracy: 1.0000 - val_loss: 0.0010 - val_accuracy: 1.0000
Epoch 26/30
50/50 [==============================] - 1s 26ms/step - loss: 2.6264e-04 - accuracy: 1.0000 - val_loss: 8.7627e-04 - val_accuracy: 1.0000
Epoch 27/30
50/50 [==============================] - 1s 26ms/step - loss: 2.3633e-04 - accuracy: 1.0000 - val_loss: 8.1398e-04 - val_accuracy: 1.0000
Epoch 28/30
50/50 [==============================] - 1s 27ms/step - loss: 2.1362e-04 - accuracy: 1.0000 - val_loss: 7.5192e-04 - val_accuracy: 1.0000
Epoch 29/30
50/50 [==============================] - 1s 26ms/step - loss: 1.9402e-04 - accuracy: 1.0000 - val_loss: 6.9469e-04 - val_accuracy: 1.0000
Epoch 30/30
50/50 [==============================] - 1s 27ms/step - loss: 1.7712e-04 - accuracy: 1.0000 - val_loss: 6.3455e-04 - val_accuracy: 1.0000
  → ステージ1 完了: 30エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 1 (1桁)] 汎化テスト（未学習20件）: 20/20 = 100.0%
  　正解例（最大5件）:
    Q:            2+4  正解: 6        予測: 6        〇
    Q:            1+9  正解: 10       予測: 10       〇
    Q:            5+9  正解: 14       予測: 14       〇
    Q:            4+2  正解: 6        予測: 6        〇
    Q:            1+5  正解: 6        予測: 6        〇
  ★ 全100通り正解率: 100/100=100.0%
  （このステージのホールドアウト20件は次ステージの訓練に合流予定）

--- Stage 2: 2桁  サンプル数=10000（うち20%ホールドアウト）---
  （前ステージのホールドアウト20件をこのステージの訓練に合流）
  訓練:7426件  バリデーション:1000件  このステージのホールドアウト:1000件
Epoch 1/200
 5/50 [==>...........................] - ETA: 1s - loss: 2.6294 - accuracy: 0.7420WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0244s vs `on_train_batch_end` time: 0.0273s). Check your callbacks.
50/50 [==============================] - 2s 31ms/step - loss: 0.8477 - accuracy: 0.7879 - val_loss: 0.6668 - val_accuracy: 0.7742
Epoch 2/200
50/50 [==============================] - 1s 30ms/step - loss: 0.5237 - accuracy: 0.8135 - val_loss: 0.6548 - val_accuracy: 0.7855
Epoch 3/200
50/50 [==============================] - 1s 29ms/step - loss: 0.4755 - accuracy: 0.8262 - val_loss: 0.5973 - val_accuracy: 0.7869
Epoch 4/200
50/50 [==============================] - 1s 30ms/step - loss: 0.4567 - accuracy: 0.8325 - val_loss: 0.5484 - val_accuracy: 0.7940
Epoch 5/200
50/50 [==============================] - 1s 30ms/step - loss: 0.4318 - accuracy: 0.8358 - val_loss: 0.5357 - val_accuracy: 0.7984
Epoch 6/200
50/50 [==============================] - 1s 30ms/step - loss: 0.4270 - accuracy: 0.8358 - val_loss: 0.5057 - val_accuracy: 0.8046
Epoch 7/200
50/50 [==============================] - 1s 29ms/step - loss: 0.3793 - accuracy: 0.8510 - val_loss: 0.5261 - val_accuracy: 0.8051
Epoch 8/200
50/50 [==============================] - 1s 30ms/step - loss: 0.3524 - accuracy: 0.8581 - val_loss: 0.4651 - val_accuracy: 0.8274
Epoch 9/200
50/50 [==============================] - 1s 30ms/step - loss: 0.2936 - accuracy: 0.8808 - val_loss: 0.5328 - val_accuracy: 0.8104
Epoch 10/200
50/50 [==============================] - 1s 30ms/step - loss: 0.2676 - accuracy: 0.8910 - val_loss: 0.4317 - val_accuracy: 0.8411

...

Epoch 191/200
50/50 [==============================] - 2s 30ms/step - loss: 2.3071e-05 - accuracy: 1.0000 - val_loss: 3.7449e-04 - val_accuracy: 0.9999
Epoch 192/200
50/50 [==============================] - 1s 30ms/step - loss: 2.2215e-05 - accuracy: 1.0000 - val_loss: 4.0089e-04 - val_accuracy: 0.9998
Epoch 193/200
50/50 [==============================] - 2s 30ms/step - loss: 2.1997e-05 - accuracy: 1.0000 - val_loss: 3.9311e-04 - val_accuracy: 0.9998
Epoch 194/200
50/50 [==============================] - 2s 31ms/step - loss: 2.1001e-05 - accuracy: 1.0000 - val_loss: 3.4882e-04 - val_accuracy: 0.9999
Epoch 195/200
50/50 [==============================] - 2s 31ms/step - loss: 2.0668e-05 - accuracy: 1.0000 - val_loss: 3.5812e-04 - val_accuracy: 0.9998
Epoch 196/200
50/50 [==============================] - 2s 31ms/step - loss: 1.9701e-05 - accuracy: 1.0000 - val_loss: 4.1033e-04 - val_accuracy: 0.9998
Epoch 197/200
50/50 [==============================] - 2s 31ms/step - loss: 1.9224e-05 - accuracy: 1.0000 - val_loss: 3.2137e-04 - val_accuracy: 0.9999
Epoch 198/200
50/50 [==============================] - 1s 30ms/step - loss: 1.8851e-05 - accuracy: 1.0000 - val_loss: 3.6300e-04 - val_accuracy: 0.9999
Epoch 199/200
50/50 [==============================] - 2s 31ms/step - loss: 1.8111e-05 - accuracy: 1.0000 - val_loss: 3.5495e-04 - val_accuracy: 0.9999
Epoch 200/200
50/50 [==============================] - 2s 30ms/step - loss: 1.8034e-05 - accuracy: 1.0000 - val_loss: 3.5005e-04 - val_accuracy: 0.9999
  → ステージ2 完了: 200エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 2 (2桁)] 汎化テスト（未学習500件）: 500/500 = 100.0%
  　正解例（最大5件）:
    Q:          97+63  正解: 160      予測: 160      〇
    Q:           0+86  正解: 86       予測: 86       〇
    Q:           2+56  正解: 58       予測: 58       〇
    Q:           8+77  正解: 85       予測: 85       〇
    Q:           8+62  正解: 70       予測: 70       〇
  （このステージのホールドアウト1000件は次ステージの訓練に合流予定）

--- Stage 3: 3桁  サンプル数=40000（うち20%ホールドアウト）---
  （前ステージのホールドアウト1000件をこのステージの訓練に合流）
  訓練:33000件  バリデーション:4000件  このステージのホールドアウト:4000件
Epoch 1/200
 5/50 [==>...........................] - ETA: 1s - loss: 3.9319 - accuracy: 0.7186WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0224s vs `on_train_batch_end` time: 0.0228s). Check your callbacks.
50/50 [==============================] - 2s 42ms/step - loss: 1.2845 - accuracy: 0.6953 - val_loss: 0.9081 - val_accuracy: 0.6865
Epoch 2/200
50/50 [==============================] - 2s 42ms/step - loss: 0.8530 - accuracy: 0.7082 - val_loss: 0.8511 - val_accuracy: 0.6977
Epoch 3/200
50/50 [==============================] - 2s 42ms/step - loss: 0.8075 - accuracy: 0.7138 - val_loss: 0.8102 - val_accuracy: 0.7032
Epoch 4/200
50/50 [==============================] - 2s 42ms/step - loss: 0.7528 - accuracy: 0.7250 - val_loss: 0.7963 - val_accuracy: 0.7075
Epoch 5/200
50/50 [==============================] - 2s 42ms/step - loss: 0.7259 - accuracy: 0.7298 - val_loss: 0.7753 - val_accuracy: 0.7112
Epoch 6/200
50/50 [==============================] - 2s 41ms/step - loss: 0.7069 - accuracy: 0.7332 - val_loss: 0.7922 - val_accuracy: 0.7086
Epoch 7/200
50/50 [==============================] - 2s 41ms/step - loss: 0.6743 - accuracy: 0.7448 - val_loss: 0.7235 - val_accuracy: 0.7249
Epoch 8/200
50/50 [==============================] - 2s 41ms/step - loss: 0.6405 - accuracy: 0.7540 - val_loss: 0.7054 - val_accuracy: 0.7272
Epoch 9/200
50/50 [==============================] - 2s 41ms/step - loss: 0.6107 - accuracy: 0.7637 - val_loss: 0.6624 - val_accuracy: 0.7467
Epoch 10/200
50/50 [==============================] - 2s 42ms/step - loss: 0.5731 - accuracy: 0.7785 - val_loss: 0.6110 - val_accuracy: 0.7693

...

Epoch 191/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 0.9999
Epoch 192/200
50/50 [==============================] - 2s 41ms/step - loss: 9.8760e-04 - accuracy: 1.0000 - val_loss: 0.0013 - val_accuracy: 0.9999
Epoch 193/200
50/50 [==============================] - 2s 41ms/step - loss: 9.9786e-04 - accuracy: 1.0000 - val_loss: 0.0012 - val_accuracy: 1.0000
Epoch 194/200
50/50 [==============================] - 2s 41ms/step - loss: 9.2643e-04 - accuracy: 1.0000 - val_loss: 0.0011 - val_accuracy: 1.0000
Epoch 195/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0012 - accuracy: 0.9998 - val_loss: 0.0013 - val_accuracy: 0.9999
Epoch 196/200
50/50 [==============================] - 2s 41ms/step - loss: 8.7761e-04 - accuracy: 1.0000 - val_loss: 0.0010 - val_accuracy: 0.9999
Epoch 197/200
50/50 [==============================] - 2s 41ms/step - loss: 7.1170e-04 - accuracy: 1.0000 - val_loss: 8.9788e-04 - val_accuracy: 1.0000
Epoch 198/200
50/50 [==============================] - 2s 41ms/step - loss: 6.6912e-04 - accuracy: 1.0000 - val_loss: 8.9157e-04 - val_accuracy: 1.0000
Epoch 199/200
50/50 [==============================] - 2s 41ms/step - loss: 6.2324e-04 - accuracy: 1.0000 - val_loss: 8.4495e-04 - val_accuracy: 1.0000
Epoch 200/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0020 - accuracy: 0.9994 - val_loss: 0.0028 - val_accuracy: 0.9992
  → ステージ3 完了: 200エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 3 (3桁)] 汎化テスト（未学習500件）: 499/500 = 99.8%
  　正解例（最大5件）:
    Q:          6+192  正解: 198      予測: 198      〇
    Q:          5+122  正解: 127      予測: 127      〇
    Q:          779+5  正解: 784      予測: 784      〇
    Q:         557+60  正解: 617      予測: 617      〇
    Q:         895+63  正解: 958      予測: 958      〇
  　不正解例（最大5件）:
    Q:        990+100  正解: 1090     予測: 1000     ×
  （このステージのホールドアウト4000件は次ステージの訓練に合流予定）

--- Stage 4: 4桁  サンプル数=40000（うち20%ホールドアウト）---
  （前ステージのホールドアウト4000件をこのステージの訓練に合流）
  訓練:36000件  バリデーション:4000件  このステージのホールドアウト:4000件
Epoch 1/200
 6/50 [==>...........................] - ETA: 2s - loss: 2.7236 - accuracy: 0.7912WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0275s vs `on_train_batch_end` time: 0.0333s). Check your callbacks.
50/50 [==============================] - 2s 44ms/step - loss: 1.0343 - accuracy: 0.7669 - val_loss: 0.5554 - val_accuracy: 0.7990
Epoch 2/200
50/50 [==============================] - 2s 42ms/step - loss: 0.3871 - accuracy: 0.8650 - val_loss: 0.3965 - val_accuracy: 0.8581
Epoch 3/200
50/50 [==============================] - 2s 41ms/step - loss: 0.3003 - accuracy: 0.8953 - val_loss: 0.3300 - val_accuracy: 0.8803
Epoch 4/200
50/50 [==============================] - 2s 41ms/step - loss: 0.2323 - accuracy: 0.9193 - val_loss: 0.2683 - val_accuracy: 0.9037
Epoch 5/200
50/50 [==============================] - 2s 41ms/step - loss: 0.2099 - accuracy: 0.9251 - val_loss: 0.2471 - val_accuracy: 0.9098
Epoch 6/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1846 - accuracy: 0.9325 - val_loss: 0.2036 - val_accuracy: 0.9270
Epoch 7/200
50/50 [==============================] - 2s 41ms/step - loss: 0.2218 - accuracy: 0.9165 - val_loss: 0.2614 - val_accuracy: 0.8998
Epoch 8/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1515 - accuracy: 0.9459 - val_loss: 0.1620 - val_accuracy: 0.9428
Epoch 9/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1202 - accuracy: 0.9576 - val_loss: 0.1351 - val_accuracy: 0.9543
Epoch 10/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0991 - accuracy: 0.9673 - val_loss: 0.1217 - val_accuracy: 0.9579

...

Epoch 191/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0015 - accuracy: 0.9998 - val_loss: 0.0040 - val_accuracy: 0.9989
Epoch 192/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0013 - accuracy: 0.9999 - val_loss: 0.0048 - val_accuracy: 0.9984
Epoch 193/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0016 - accuracy: 0.9998 - val_loss: 0.0042 - val_accuracy: 0.9990
Epoch 194/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0016 - accuracy: 0.9997 - val_loss: 0.0034 - val_accuracy: 0.9991
Epoch 195/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0011 - accuracy: 0.9998 - val_loss: 0.0028 - val_accuracy: 0.9994
Epoch 196/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0011 - accuracy: 0.9998 - val_loss: 0.0032 - val_accuracy: 0.9992
Epoch 197/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0011 - accuracy: 0.9998 - val_loss: 0.0036 - val_accuracy: 0.9992
Epoch 198/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0011 - accuracy: 0.9999 - val_loss: 0.0028 - val_accuracy: 0.9993
Epoch 199/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0015 - accuracy: 0.9996 - val_loss: 0.0058 - val_accuracy: 0.9980
Epoch 200/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0029 - accuracy: 0.9992 - val_loss: 0.0065 - val_accuracy: 0.9979
  → ステージ4 完了: 200エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 4 (4桁)] 汎化テスト（未学習500件）: 495/500 = 99.0%
  　正解例（最大5件）:
    Q:          3+401  正解: 404      予測: 404      〇
    Q:           55+0  正解: 55       予測: 55       〇
    Q:          80+89  正解: 169      予測: 169      〇
    Q:          78+21  正解: 99       予測: 99       〇
    Q:           7+28  正解: 35       予測: 35       〇
  　不正解例（最大5件）:
    Q:      2815+1551  正解: 4366     予測: 3366     ×
    Q:      3085+8148  正解: 11233    予測: 12233    ×
    Q:      1303+1101  正解: 2404     予測: 1504     ×
    Q:      7899+2891  正解: 10790    予測: 10880    ×
    Q:      8854+6764  正解: 15618    予測: 16618    ×
  （このステージのホールドアウト4000件は次ステージの訓練に合流予定）

--- Stage 5: 5桁  サンプル数=40000（うち20%ホールドアウト）---
  （前ステージのホールドアウト4000件をこのステージの訓練に合流）
  訓練:36000件  バリデーション:4000件  このステージのホールドアウト:4000件
Epoch 1/200
 5/50 [==>...........................] - ETA: 1s - loss: 1.6160 - accuracy: 0.8543WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0236s vs `on_train_batch_end` time: 0.0262s). Check your callbacks.
50/50 [==============================] - 2s 43ms/step - loss: 0.6489 - accuracy: 0.8573 - val_loss: 0.3283 - val_accuracy: 0.8833
Epoch 2/200
50/50 [==============================] - 2s 42ms/step - loss: 0.2070 - accuracy: 0.9279 - val_loss: 0.2177 - val_accuracy: 0.9248
Epoch 3/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1618 - accuracy: 0.9445 - val_loss: 0.1852 - val_accuracy: 0.9357
Epoch 4/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1415 - accuracy: 0.9523 - val_loss: 0.1655 - val_accuracy: 0.9428
Epoch 5/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1244 - accuracy: 0.9573 - val_loss: 0.1522 - val_accuracy: 0.9465
Epoch 6/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1153 - accuracy: 0.9601 - val_loss: 0.1416 - val_accuracy: 0.9518
Epoch 7/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1079 - accuracy: 0.9626 - val_loss: 0.1342 - val_accuracy: 0.9530
Epoch 8/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1017 - accuracy: 0.9646 - val_loss: 0.1304 - val_accuracy: 0.9545
Epoch 9/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0944 - accuracy: 0.9680 - val_loss: 0.1200 - val_accuracy: 0.9584
Epoch 10/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0886 - accuracy: 0.9692 - val_loss: 0.1145 - val_accuracy: 0.9606

...

Epoch 191/200
50/50 [==============================] - 2s 41ms/step - loss: 7.3573e-04 - accuracy: 1.0000 - val_loss: 0.0111 - val_accuracy: 0.9966
Epoch 192/200
50/50 [==============================] - 2s 42ms/step - loss: 6.5043e-04 - accuracy: 1.0000 - val_loss: 0.0111 - val_accuracy: 0.9964
Epoch 193/200
50/50 [==============================] - 2s 41ms/step - loss: 6.8332e-04 - accuracy: 1.0000 - val_loss: 0.0118 - val_accuracy: 0.9959
Epoch 194/200
50/50 [==============================] - 2s 41ms/step - loss: 8.3552e-04 - accuracy: 0.9999 - val_loss: 0.0117 - val_accuracy: 0.9958
Epoch 195/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0010 - accuracy: 0.9999 - val_loss: 0.0145 - val_accuracy: 0.9946
Epoch 196/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0061 - accuracy: 0.9980 - val_loss: 0.0342 - val_accuracy: 0.9896
Epoch 197/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0968 - accuracy: 0.9751 - val_loss: 0.2540 - val_accuracy: 0.9327
Epoch 198/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1228 - accuracy: 0.9611 - val_loss: 0.0598 - val_accuracy: 0.9802
Epoch 199/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0207 - accuracy: 0.9939 - val_loss: 0.0283 - val_accuracy: 0.9910
Epoch 200/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0084 - accuracy: 0.9979 - val_loss: 0.0232 - val_accuracy: 0.9919
  → ステージ5 完了: 200エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 5 (5桁)] 汎化テスト（未学習500件）: 484/500 = 96.8%
  　正解例（最大5件）:
    Q:        670+443  正解: 1113     予測: 1113     〇
    Q:        783+619  正解: 1402     予測: 1402     〇
    Q:          7+918  正解: 925      予測: 925      〇
    Q:      6050+6759  正解: 12809    予測: 12809    〇
    Q:         8+3142  正解: 3150     予測: 3150     〇
  　不正解例（最大5件）:
    Q:     5692+98421  正解: 104113   予測: 103113   ×
    Q:     7021+79269  正解: 86290    予測: 85290    ×
    Q:    82355+18683  正解: 101038   予測: 111038   ×
    Q:    74889+78770  正解: 153659   予測: 155659   ×
    Q:    22712+61772  正解: 84484    予測: 85484    ×
  （このステージのホールドアウト4000件は次ステージの訓練に合流予定）

--- Stage 6: 6桁  サンプル数=40000（うち20%ホールドアウト）---
  （前ステージのホールドアウト4000件をこのステージの訓練に合流）
  訓練:36000件  バリデーション:4000件  このステージのホールドアウト:4000件
Epoch 1/200
 5/50 [==>...........................] - ETA: 1s - loss: 1.2452 - accuracy: 0.8738WARNING:tensorflow:Callback method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0160s vs `on_train_batch_end` time: 0.0223s). Check your callbacks.
50/50 [==============================] - 2s 41ms/step - loss: 0.4905 - accuracy: 0.8953 - val_loss: 0.2754 - val_accuracy: 0.9081
Epoch 2/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1741 - accuracy: 0.9395 - val_loss: 0.1983 - val_accuracy: 0.9296
Epoch 3/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1401 - accuracy: 0.9510 - val_loss: 0.1774 - val_accuracy: 0.9348
Epoch 4/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1313 - accuracy: 0.9536 - val_loss: 0.1536 - val_accuracy: 0.9438
Epoch 5/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1181 - accuracy: 0.9571 - val_loss: 0.1467 - val_accuracy: 0.9466
Epoch 6/200
50/50 [==============================] - 2s 41ms/step - loss: 0.1092 - accuracy: 0.9611 - val_loss: 0.1365 - val_accuracy: 0.9496
Epoch 7/200
50/50 [==============================] - 2s 42ms/step - loss: 0.1002 - accuracy: 0.9645 - val_loss: 0.1271 - val_accuracy: 0.9532
Epoch 8/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0904 - accuracy: 0.9674 - val_loss: 0.1223 - val_accuracy: 0.9547
Epoch 9/200
50/50 [==============================] - 2s 42ms/step - loss: 0.0853 - accuracy: 0.9703 - val_loss: 0.1153 - val_accuracy: 0.9586
Epoch 10/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0812 - accuracy: 0.9710 - val_loss: 0.1116 - val_accuracy: 0.9590

...

Epoch 191/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0052 - accuracy: 0.9990 - val_loss: 0.0497 - val_accuracy: 0.9851
Epoch 192/200
50/50 [==============================] - 2s 40ms/step - loss: 0.0038 - accuracy: 0.9993 - val_loss: 0.0483 - val_accuracy: 0.9855
Epoch 193/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0031 - accuracy: 0.9996 - val_loss: 0.0464 - val_accuracy: 0.9862
Epoch 194/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0031 - accuracy: 0.9996 - val_loss: 0.0473 - val_accuracy: 0.9857
Epoch 195/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0022 - accuracy: 0.9998 - val_loss: 0.0452 - val_accuracy: 0.9866
Epoch 196/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0023 - accuracy: 0.9998 - val_loss: 0.0466 - val_accuracy: 0.9865
Epoch 197/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0018 - accuracy: 0.9999 - val_loss: 0.0454 - val_accuracy: 0.9867
Epoch 198/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0445 - val_accuracy: 0.9873
Epoch 199/200
50/50 [==============================] - 2s 41ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0437 - val_accuracy: 0.9876
Epoch 200/200
50/50 [==============================] - 2s 41ms/step - loss: 9.4748e-04 - accuracy: 1.0000 - val_loss: 0.0445 - val_accuracy: 0.9871
  → ステージ6 完了: 200エポック実行
  （val_lossが最良だったエポックの重みに復元しました）

  ◆ [Stage 6 (6桁)] 汎化テスト（未学習500件）: 464/500 = 92.8%
  　正解例（最大5件）:
    Q:   32397+566239  正解: 598636   予測: 598636   〇
    Q:     2464+28142  正解: 30606    予測: 30606    〇
    Q:       6850+459  正解: 7309     予測: 7309     〇
    Q:         1+2899  正解: 2900     予測: 2900     〇
    Q:       5280+897  正解: 6177     予測: 6177     〇
  　不正解例（最大5件）:
    Q:     613+369389  正解: 370002   予測: 360002   ×
    Q:    4131+195271  正解: 199402   予測: 109402   ×
    Q:  317371+227535  正解: 544906   予測: 465906   ×
    Q:  257531+983972  正解: 1241503  予測: 1340503  ×
    Q:    9639+888765  正解: 898404   予測: 897404   ×
  （最終ステージのホールドアウト4000件は最終テスト用として保持・以降も学習させません）

============================================================
最終汎化テスト（6桁の未学習データのみで評価）
============================================================

  ◆ [最終（6桁・未学習データ）] 汎化テスト（未学習1000件）: 933/1000 = 93.3%
  　正解例（最大10件）:
    Q:       121+5669  正解: 5790     予測: 5790     〇
    Q:        0+25725  正解: 25725    予測: 25725    〇
    Q:     621+453680  正解: 454301   予測: 454301   〇
    Q:        713+504  正解: 1217     予測: 1217     〇
    Q:      66+306305  正解: 306371   予測: 306371   〇
    Q:         17+334  正解: 351      予測: 351      〇
    Q:     723+754534  正解: 755257   予測: 755257   〇
    Q:           5+68  正解: 73       予測: 73       〇
    Q:      141343+52  正解: 141395   予測: 141395   〇
    Q:         712+74  正解: 786      予測: 786      〇
  　不正解例（最大10件）:
    Q:  258576+410026  正解: 668602   予測: 658602   ×
    Q:   76603+938587  正解: 1015190  予測: 1055190  ×
    Q:   96376+253043  正解: 349419   予測: 329419   ×
    Q:  476883+255929  正解: 732812   予測: 812812   ×
    Q:    41257+20836  正解: 62093    予測: 52093    ×
    Q:  264791+959570  正解: 1224361  予測: 1104361  ×
    Q:    69860+48923  正解: 118783   予測: 108783   ×
    Q:  658779+455153  正解: 1113932  予測: 1143932  ×
    Q:  668799+948447  正解: 1617246  予測: 1506246  ×
    Q:  489594+490359  正解: 979953   予測: 909953   ×

============================================================
汎化スコアまとめ（ステージごと）
============================================================
  Stage 1 (1桁): 100.0% ████████████████████
  Stage 2 (2桁): 100.0% ████████████████████
  Stage 3 (3桁):  99.8% ███████████████████
  Stage 4 (4桁):  99.0% ███████████████████
  Stage 5 (5桁):  96.8% ███████████████████
  Stage 6 (6桁):  92.8% ██████████████████

  最終（6桁・一度も学習していないデータ）: 93.3%
============================================================

(tf_gpu_env) C:\Users\user\Keras>
```
