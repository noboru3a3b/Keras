"""
lstm_add_seq2seq9.py
 ‐ カリキュラム学習版 Seq2Seq による整数加算（汎化確認強化版 v4）
 ‐ 前バージョン(lstm_add_seq2seq8.py / v3)からの主な変更点：
     [ユーザー指示1] a<=b の制約を撤廃し、全ての順序対 (a, b) を生成する
                     → 例えば「3+9」だけでなく「9+3」も別データとして学習・評価対象にする
     [ユーザー指示2] あるステージのホールドアウト（未学習にしていた分）は、
                     次のステージが始まる時点で訓練データに合流させる
                     → どの桁も最終的には「一度も見ていないデータ」を残したまま次に進まない
     [ユーザー指示3] 上記[2]の結果、最終ステージ（6桁）のホールドアウトだけが
                     最後まで一度もモデルに見せられないデータとして残る
                     → 最終汎化テストはこの「6桁の未学習データ」のみで行う
     [v4] 各ステージの学習後半で val_loss が悪化する（過学習）ケースが
          観測されたため、ModelCheckpoint(save_best_only=True) で
          「そのステージ内でval_lossが最良だったエポックの重み」を保存し、
          model.fit() 終了直後に model.load_weights() で読み込み直す。
          あわせて、これまで定義だけされていて実際には使われていなかった
          EarlyStoppingAfterMinEpochs も有効化する
          （無駄なエポックを削り、学習時間を節約する目的）。
"""

# ============================================================
# 0. Imports
# ============================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 1. Hyper-parameters
# ============================================================
MAX_DIGITS             = 6
REVERSE                = True

#LATENT_DIM             = 256
LATENT_DIM             = 512

#STAGE_SAMPLES_MAX      = 20_000
STAGE_SAMPLES_MAX      = 40_000

BATCH_SIZE             = 128
REPLAY_RATIO           = 0.3

#MAX_EPOCHS             = 30
MAX_EPOCHS             = [30, 200, 200, 200, 200, 200]

MIN_EPOCHS             = 20

#PATIENCE               = 20
PATIENCE               = 60

TARGET_STEPS_PER_EPOCH = 50
MIN_SAMPLES_FOR_SPLIT  = 500   # この件数未満なら全通り生成

# [v4] 各ステージのベスト重み（val_loss最良時点）を一時保存するファイル
#      ステージごとに上書きされるので、ステージをまたいで蓄積はしない
CHECKPOINT_PATH        = "stage_best.weights.h5"

# ★ 汎化確認用: 各ステージで生成したうちこの割合を「未学習テスト」用に確保
#    （ただし[ユーザー指示2]により、次ステージ開始時に訓練へ合流させる）
HOLDOUT_RATIO          = 0.2   # 20%をホールドアウト
#HOLDOUT_MIN            = 200   # ホールドアウトの最低件数
HOLDOUT_MIN            = 10   # ホールドアウトの最低件数

# ============================================================
# 2. Character table
# ============================================================
CHARS = "0123456789+ "

class CharacterTable:
    def __init__(self, chars: str):
        self.chars        = sorted(set(chars))
        self.char_indices = {c: i for i, c in enumerate(self.chars)}
        self.indices_char = {i: c for i, c in enumerate(self.chars)}
        self.num_chars    = len(self.chars)
        self.pad_idx      = self.char_indices[" "]

    def encode(self, s: str, length: int) -> np.ndarray:
        x = np.zeros((length, self.num_chars), dtype=np.float32)
        for i, c in enumerate(s[:length]):
            x[i, self.char_indices[c]] = 1.0
        return x

    def decode(self, x: np.ndarray, calc_argmax: bool = True) -> str:
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[int(i)] for i in x)

ctable        = CharacterTable(CHARS)
NUM_CHARS     = ctable.num_chars
PAD           = " "

GLOBAL_MAXLEN = MAX_DIGITS + 1 + MAX_DIGITS
GLOBAL_ANS_LEN = MAX_DIGITS + 2

# ============================================================
# 3. Utility
# ============================================================
def repeat_to_target(enc, dec_in, dec_tgt, target_size):
    n = len(enc)
    if n >= target_size:
        return enc, dec_in, dec_tgt
    k = target_size // n + 1
    enc     = np.concatenate([enc]     * k)[:target_size]
    dec_in  = np.concatenate([dec_in]  * k)[:target_size]
    dec_tgt = np.concatenate([dec_tgt] * k)[:target_size]
    idx = np.random.permutation(target_size)
    return enc[idx], dec_in[idx], dec_tgt[idx]

# ============================================================
# 4. Data generation
# ============================================================
def _rand_number(digits: int) -> int:
    n = np.random.randint(1, digits + 1)
    return int("".join(np.random.choice(list("0123456789")) for _ in range(n)))

def _build_tensors(pairs):
    """
    pairs: list of (q_padded, ans_padded) の文字列タプル
    戻り値: enc_in, dec_in, dec_tgt
    """
    N = len(pairs)
    enc_in  = np.zeros((N, GLOBAL_MAXLEN,   NUM_CHARS), dtype=np.float32)
    dec_in  = np.zeros((N, GLOBAL_ANS_LEN,  NUM_CHARS), dtype=np.float32)
    dec_tgt = np.zeros((N, GLOBAL_ANS_LEN,  NUM_CHARS), dtype=np.float32)
    for i, (q, ans) in enumerate(pairs):
        enc_in[i]  = ctable.encode(q, GLOBAL_MAXLEN)
        ans_enc    = ctable.encode(ans, GLOBAL_ANS_LEN)
        dec_tgt[i] = ans_enc
        bos = np.zeros(NUM_CHARS, dtype=np.float32)
        bos[ctable.pad_idx] = 1.0
        dec_in[i, 0]  = bos
        dec_in[i, 1:] = ans_enc[:-1]
    return enc_in, dec_in, dec_tgt

def _make_pair(a, b, reverse):
    q_raw  = f"{a}+{b}"
    ans_raw = str(a + b)
    q_pad   = (q_raw[::-1] if reverse else q_raw).ljust(GLOBAL_MAXLEN, PAD)
    ans_pad = ans_raw.ljust(GLOBAL_ANS_LEN, PAD)
    return q_pad, ans_pad, q_raw  # q_raw はデバッグ表示用

def generate_all_combinations(digits, reverse=True):
    """
    全ペアを列挙して返す。小桁（現状は1桁のみ）専用。

    [ユーザー指示1] a<=b の制約を撤廃。
    以前は for b in range(a, max_val) として a<=b の組しか生成していなかった
    （「3+9」はあっても「9+3」は無い、という非対称なデータだった）。
    今回は全ての順序対 (a, b) を生成するため、件数は
      以前: max_val*(max_val+1)/2 通り（三角数）
      現在: max_val**2         通り（正方数）
    に増える。
    """
    max_val = 10 ** digits
    pairs, q_raws = [], []
    for a in range(max_val):
        for b in range(max_val):
            q_pad, ans_pad, q_raw = _make_pair(a, b, reverse)
            pairs.append((q_pad, ans_pad))
            q_raws.append(q_raw)
    enc, dec_in, dec_tgt = _build_tensors(pairs)
    perm = np.random.permutation(len(pairs))
    return enc[perm], dec_in[perm], dec_tgt[perm], [q_raws[i] for i in perm]

# ============================================================
# [変更1] ホールドアウト分離付きデータ生成
# ============================================================
def generate_data_with_holdout(num_samples, digits, reverse=True, seed=None):
    """
    num_samples 件をランダム生成し、
    ・訓練用 (1 - HOLDOUT_RATIO)
    ・ホールドアウト用 (HOLDOUT_RATIO) ← このステージでは訓練に使わない
      （[ユーザー指示2]により、次ステージ開始時に訓練へ合流する）
    に分割して返す。

    [ユーザー指示1] 重複判定のキーを (a, b) の順序付きペアに変更。
    以前は key=(min(a,b), max(a,b)) として順序を無視していたため、
    「3+9」を生成すると「9+3」は同じキー扱いで別ペアとして生成され得なかった。
    今回は順序も区別するペアとして扱う。
    """
    if seed is not None:
        np.random.seed(seed)

    seen = set()
    train_pairs, hold_pairs = [], []
    train_q_raws, hold_q_raws = [], []

    n_hold = max(HOLDOUT_MIN, int(num_samples * HOLDOUT_RATIO))
    n_train = num_samples - n_hold

    # まずホールドアウト用を確保（未使用ペア）
    attempts = 0
    while len(hold_pairs) < n_hold and attempts < num_samples * 10:
        a, b = _rand_number(digits), _rand_number(digits)
        key = (a, b)
        if key not in seen:
            seen.add(key)
            q_pad, ans_pad, q_raw = _make_pair(a, b, reverse)
            hold_pairs.append((q_pad, ans_pad))
            hold_q_raws.append(q_raw)
        attempts += 1

    # 次に訓練用を確保（ホールドアウトと重複しないペア）
    attempts = 0
    while len(train_pairs) < n_train and attempts < num_samples * 10:
        a, b = _rand_number(digits), _rand_number(digits)
        key = (a, b)
        if key not in seen:
            seen.add(key)
            q_pad, ans_pad, q_raw = _make_pair(a, b, reverse)
            train_pairs.append((q_pad, ans_pad))
            train_q_raws.append(q_raw)
        attempts += 1

    enc_tr, dec_tr, tgt_tr   = _build_tensors(train_pairs)
    enc_ho, dec_ho, tgt_ho   = _build_tensors(hold_pairs)

    perm_tr = np.random.permutation(len(train_pairs))
    perm_ho = np.random.permutation(len(hold_pairs))

    return (enc_tr[perm_tr], dec_tr[perm_tr], tgt_tr[perm_tr],
            enc_ho[perm_ho], dec_ho[perm_ho], tgt_ho[perm_ho],
            [train_q_raws[i] for i in perm_tr],
            [hold_q_raws[i]  for i in perm_ho])

# ============================================================
# 5. Build model
# ============================================================
print("モデル構築中...")

encoder_inputs = keras.Input(shape=(None, NUM_CHARS), name="encoder_input")
encoder_lstm   = layers.LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, h, c        = encoder_lstm(encoder_inputs)
enc_states     = [h, c]

decoder_inputs = keras.Input(shape=(None, NUM_CHARS), name="decoder_input")
decoder_lstm   = layers.LSTM(LATENT_DIM, return_sequences=True,
                             return_state=True, name="decoder_lstm")
dec_out, _, _  = decoder_lstm(decoder_inputs, initial_state=enc_states)
decoder_dense  = layers.Dense(NUM_CHARS, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(dec_out)

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs,
                    name="seq2seq_train")
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Inference sub-models
inf_encoder = keras.Model(encoder_inputs, enc_states, name="inf_encoder")
inf_dec_in  = keras.Input(shape=(1, NUM_CHARS))
state_h_in  = keras.Input(shape=(LATENT_DIM,))
state_c_in  = keras.Input(shape=(LATENT_DIM,))
dec_out2, h2, c2 = decoder_lstm(inf_dec_in, initial_state=[state_h_in, state_c_in])
dec_out2 = decoder_dense(dec_out2)
inf_decoder = keras.Model([inf_dec_in, state_h_in, state_c_in],
                          [dec_out2, h2, c2], name="inf_decoder")

def decode_sequence(enc_seq):
    h, c = inf_encoder.predict(enc_seq, verbose=0)
    tgt  = np.zeros((1, 1, NUM_CHARS), dtype=np.float32)
    tgt[0, 0, ctable.pad_idx] = 1.0
    out_seq = []
    for _ in range(GLOBAL_ANS_LEN):
        y, h, c = inf_decoder.predict([tgt, h, c], verbose=0)
        idx = np.argmax(y[0, 0])
        out_seq.append(ctable.indices_char[idx])
        tgt[:] = 0.0
        tgt[0, 0, idx] = 1.0
    return "".join(out_seq)

def batch_decode(enc_batch):
    """複数の enc_seq をまとめて推論（高速化版）"""
    N = len(enc_batch)
    h_batch, c_batch = inf_encoder.predict(enc_batch, verbose=0, batch_size=128)
    
    tgt = np.zeros((N, 1, NUM_CHARS), dtype=np.float32)
    tgt[:, 0, ctable.pad_idx] = 1.0
    
    results = [[] for _ in range(N)]
    for _ in range(GLOBAL_ANS_LEN):
        y, h_batch, c_batch = inf_decoder.predict(
            [tgt, h_batch, c_batch], verbose=0, batch_size=128)
        idx = np.argmax(y[:, 0, :], axis=-1)
        for i in range(N):
            results[i].append(ctable.indices_char[idx[i]])
        tgt[:] = 0.0
        tgt[np.arange(N), 0, idx] = 1.0
    
    return ["".join(r) for r in results]

# ============================================================
# [変更2] 汎化評価関数
# ============================================================
def evaluate_generalization(enc_ho, dec_ho, tgt_ho, q_raws_ho,
                            stage_label, n_show=10, n_eval_max=500):
    n = min(len(enc_ho), n_eval_max)
    idx_eval = np.random.choice(len(enc_ho), n, replace=False)
    
    enc_sub = enc_ho[idx_eval]
    
    # ★ バッチ推論に変更（1件ずつループしない）
    preds = batch_decode(enc_sub)
    
    correct = 0
    wrong_examples = []
    for j, i in enumerate(idx_eval):
        pred   = preds[j].strip()
        answer = ctable.decode(tgt_ho[i]).strip()
        q_disp = q_raws_ho[i] if q_raws_ho else "?"
        if pred == answer:
            correct += 1
        else:
            wrong_examples.append((q_disp, answer, pred))
    
    acc = correct / n * 100
    print(f"\n  ◆ [{stage_label}] 汎化テスト（未学習{n}件）: "
          f"{correct}/{n} = {acc:.1f}%")
    
    # 正解例
    print(f"  　正解例（最大{n_show}件）:")
    cnt = 0
    for j, i in enumerate(idx_eval):
        if cnt >= n_show: break
        pred   = preds[j].strip()
        answer = ctable.decode(tgt_ho[i]).strip()
        q_disp = q_raws_ho[i] if q_raws_ho else "?"
        if pred == answer:
            print(f"    Q: {q_disp:>14}  正解: {answer:<7}  予測: {pred:<7}  〇")
            cnt += 1
    
    # 不正解例
    if wrong_examples:
        print(f"  　不正解例（最大{n_show}件）:")
        for q_disp, answer, pred in wrong_examples[:n_show]:
            print(f"    Q: {q_disp:>14}  正解: {answer:<7}  予測: {pred:<7}  ×")
    
    return acc

# ============================================================
# 6. EarlyStopping callback
# ============================================================
class EarlyStoppingAfterMinEpochs(keras.callbacks.Callback):
    def __init__(self, min_epochs, patience):
        super().__init__()
        self.min_epochs = min_epochs
        self.patience   = patience
        self.best       = np.inf
        self.wait       = 0
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss", np.inf)
        if epoch < self.min_epochs - 1:
            self.best = min(self.best, val_loss)
            return
        if val_loss < self.best:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n  → Early stopping (epoch {epoch+1}, "
                      f"best val_loss={self.best:.4f})")
                self.model.stop_training = True

# ============================================================
# 7. Curriculum learning loop
# ============================================================
TARGET_SIZE_PER_EPOCH = TARGET_STEPS_PER_EPOCH * BATCH_SIZE

print("\n" + "="*60)
print("カリキュラム学習開始（汎化確認強化版 v4）")
print(f"  MIN_EPOCHS={MIN_EPOCHS}, MAX_EPOCHS={MAX_EPOCHS}, PATIENCE={PATIENCE}")
print(f"  HOLDOUT_RATIO={HOLDOUT_RATIO} → 各ステージの{int(HOLDOUT_RATIO*100)}%を一時的に未学習テスト用に確保")
print(f"  ※ ホールドアウトは評価後、次ステージの訓練データに合流します")
print(f"  ※ 最終ステージ（{MAX_DIGITS}桁）のホールドアウトのみ、最後まで未学習のまま最終テストに使用します")
print(f"  ※ [v4] 各ステージ終了後、val_lossが最良だったエポックの重みに復元してから次に進みます")
print("="*60)

replay_enc = replay_dec_in = replay_dec_tgt = None

# ★ 前ステージのホールドアウトのうち、次ステージの訓練に合流させる分（[ユーザー指示2]）
pending_carryover_enc = pending_carryover_dec = pending_carryover_tgt = None

# ★ 最終ステージのホールドアウト（最後まで一度も学習しないデータ）（[ユーザー指示3]）
final_holdout_enc = final_holdout_dec = final_holdout_tgt = None
final_holdout_q_raws = None

generalization_scores = []  # ステージごとの汎化スコアを記録

for stage_digits in range(1, MAX_DIGITS + 1):
    # [ユーザー指示1] a<=b の制約を撤廃したため、全順序対の総数は正方数になる
    n_combi = (10 ** stage_digits) ** 2
    use_all = n_combi < MIN_SAMPLES_FOR_SPLIT

    # ------------------------------------------------------------------
    # [変更3] データ取得：全通り or ホールドアウト分離付きランダム生成
    # ------------------------------------------------------------------
    if use_all:
        # 全ペアを列挙 → シャッフル → 前80%訓練 / 後20%ホールドアウト
        n_samples = n_combi
        print(f"\n--- Stage {stage_digits}: {stage_digits}桁  "
              f"全{n_combi}通り（全順序対、全件訓練 + 20%ホールドアウト）---")
        enc_all, dec_all, tgt_all, q_raws_all = generate_all_combinations(
            stage_digits, REVERSE)

        n_hold = max(HOLDOUT_MIN, int(n_combi * HOLDOUT_RATIO))
        n_hold = min(n_hold, n_combi - 1)

        enc_ho  = enc_all[:n_hold]
        dec_ho  = dec_all[:n_hold]
        tgt_ho  = tgt_all[:n_hold]
        q_raws_ho = q_raws_all[:n_hold]

        enc_tr_raw  = enc_all[n_hold:]
        dec_tr_raw  = dec_all[n_hold:]
        tgt_tr_raw  = tgt_all[n_hold:]

        # 訓練データが少なければ水増し
        enc_tr, dec_tr, tgt_tr = repeat_to_target(
            enc_tr_raw, dec_tr_raw, tgt_tr_raw, TARGET_SIZE_PER_EPOCH)

        # バリデーション：ホールドアウトの一部（訓練とは別物）
        enc_val, dec_val, tgt_val = enc_ho, dec_ho, tgt_ho

    else:
        n_samples = min(n_combi, STAGE_SAMPLES_MAX)
        print(f"\n--- Stage {stage_digits}: {stage_digits}桁  "
              f"サンプル数={n_samples}（うち{int(HOLDOUT_RATIO*100)}%ホールドアウト）---")

        (enc_tr, dec_tr, tgt_tr,
         enc_ho, dec_ho, tgt_ho,
         q_raws_tr, q_raws_ho) = generate_data_with_holdout(
            n_samples, stage_digits, REVERSE)

        # バリデーション：ホールドアウトの一部を使う（訓練と重複しない）
        n_val = max(10, len(enc_ho) // 2)
        enc_val, dec_val, tgt_val = enc_ho[:n_val], dec_ho[:n_val], tgt_ho[:n_val]
        # ホールドアウトの残りを汎化テスト用として使う
        enc_ho_test  = enc_ho[n_val:]
        dec_ho_test  = dec_ho[n_val:]
        tgt_ho_test  = tgt_ho[n_val:]
        q_raws_ho_test = q_raws_ho[n_val:]

    # このステージの「汎化テストに使うホールドアウト」を確定
    ho_enc_for_final    = enc_ho  if use_all else enc_ho_test
    ho_dec_for_final    = dec_ho  if use_all else dec_ho_test
    ho_tgt_for_final    = tgt_ho  if use_all else tgt_ho_test
    ho_q_raws_for_final = q_raws_ho if use_all else q_raws_ho_test

    # ------------------------------------------------------------------
    # [ユーザー指示2] 前ステージのホールドアウトをここで訓練に合流させる
    # ------------------------------------------------------------------
    if pending_carryover_enc is not None:
        n_carry = len(pending_carryover_enc)
        enc_tr = np.concatenate([enc_tr, pending_carryover_enc])
        dec_tr = np.concatenate([dec_tr, pending_carryover_dec])
        tgt_tr = np.concatenate([tgt_tr, pending_carryover_tgt])
        perm = np.random.permutation(len(enc_tr))
        enc_tr, dec_tr, tgt_tr = enc_tr[perm], dec_tr[perm], tgt_tr[perm]
        print(f"  （前ステージのホールドアウト{n_carry}件をこのステージの訓練に合流）")
        pending_carryover_enc = pending_carryover_dec = pending_carryover_tgt = None

    print(f"  訓練:{len(enc_tr)}件  バリデーション:{len(enc_val)}件  "
          f"このステージのホールドアウト:{len(ho_enc_for_final)}件")

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    if replay_enc is not None:
        n_rep = int(len(enc_tr) * REPLAY_RATIO)
        idx   = np.random.choice(len(replay_enc), n_rep, replace=True)
        enc_tr   = np.concatenate([enc_tr,  replay_enc[idx]])
        dec_tr   = np.concatenate([dec_tr,  replay_dec_in[idx]])
        tgt_tr   = np.concatenate([tgt_tr,  replay_dec_tgt[idx]])
        perm = np.random.permutation(len(enc_tr))
        enc_tr, dec_tr, tgt_tr = enc_tr[perm], dec_tr[perm], tgt_tr[perm]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    train_ds = (tf.data.Dataset
                .from_tensor_slices(((enc_tr, dec_tr), tgt_tr))
                .shuffle(len(enc_tr))
                .repeat()
                .batch(BATCH_SIZE))

    early_stop = EarlyStoppingAfterMinEpochs(MIN_EPOCHS, PATIENCE)

    # [v4] このステージ内でval_lossが最良だったエポックの重みだけをファイルに保存
    #      save_best_only=True なので、改善しなかったエポックでは上書きされない
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath          = CHECKPOINT_PATH,
        monitor           = "val_loss",
        save_best_only    = True,
        save_weights_only = True,
        verbose            = 0,
    )

    hist = model.fit(
        train_ds,
        steps_per_epoch = TARGET_STEPS_PER_EPOCH,
        validation_data = ([enc_val, dec_val], tgt_val),

#        epochs          = MAX_EPOCHS
        epochs          = MAX_EPOCHS[stage_digits - 1],

        callbacks       = [checkpoint_cb, early_stop],

        verbose         = 1
    )
    print(f"  → ステージ{stage_digits} 完了: {len(hist.history['loss'])}エポック実行")

    # [v4] 学習後半で val_loss が悪化していても、ここでベスト時点の重みに戻す。
    #      encoder_lstm / decoder_lstm / decoder_dense は inf_encoder / inf_decoder と
    #      レイヤーを共有しているため、model 側をロードし直すだけで
    #      推論用サブモデル（decode_sequence / batch_decode が使う方）にも自動的に反映される。
    model.load_weights(CHECKPOINT_PATH)
    print(f"  （val_lossが最良だったエポックの重みに復元しました）")

    # ------------------------------------------------------------------
    # [変更4] ステージ終了後の汎化テスト（このステージ内の未学習データで評価）
    # ------------------------------------------------------------------
    acc = evaluate_generalization(
        ho_enc_for_final, ho_dec_for_final, ho_tgt_for_final,
        ho_q_raws_for_final,
        stage_label=f"Stage {stage_digits} ({stage_digits}桁)",
        n_show=5
    )
    generalization_scores.append((stage_digits, acc))

    # 小桁は全通りチェックも継続
    if use_all:
        enc_all_check, _, tgt_all_check, _ = generate_all_combinations(
            stage_digits, REVERSE)
        ok = sum(
            decode_sequence(enc_all_check[np.newaxis, i]).strip()
            == ctable.decode(tgt_all_check[i]).strip()
            for i in range(len(enc_all_check))
        )
        total = len(enc_all_check)
        print(f"  ★ 全{total}通り正解率: {ok}/{total}={ok/total*100:.1f}%")

    # ------------------------------------------------------------------
    # [ユーザー指示2/3] このステージのホールドアウトの扱いを決定
    #   ・最終ステージでなければ → 次ステージの訓練に合流させる（pending化）
    #   ・最終ステージなら       → 最後まで未学習のまま最終テスト用に保持
    # ------------------------------------------------------------------
    if stage_digits < MAX_DIGITS:
        pending_carryover_enc = ho_enc_for_final
        pending_carryover_dec = ho_dec_for_final
        pending_carryover_tgt = ho_tgt_for_final
        print(f"  （このステージのホールドアウト{len(ho_enc_for_final)}件は"
              f"次ステージの訓練に合流予定）")
    else:
        final_holdout_enc    = ho_enc_for_final
        final_holdout_dec    = ho_dec_for_final
        final_holdout_tgt    = ho_tgt_for_final
        final_holdout_q_raws = ho_q_raws_for_final
        print(f"  （最終ステージのホールドアウト{len(ho_enc_for_final)}件は"
              f"最終テスト用として保持・以降も学習させません）")

    # ------------------------------------------------------------------
    # Update replay buffer
    # ------------------------------------------------------------------
    if replay_enc is None:
        replay_enc, replay_dec_in, replay_dec_tgt = enc_tr, dec_tr, tgt_tr
    else:
        replay_enc     = np.concatenate([replay_enc,     enc_tr])
        replay_dec_in  = np.concatenate([replay_dec_in,  dec_tr])
        replay_dec_tgt = np.concatenate([replay_dec_tgt, tgt_tr])

# ============================================================
# 8. 最終汎化テスト
#    [ユーザー指示3] 最終ステージ（MAX_DIGITS桁）のホールドアウトのみを使用。
#    それ以外の桁のホールドアウトは、すでに各ステージの終了時点で
#    次ステージの訓練データに合流済みのため、ここでは「未学習」ではない。
# ============================================================
print("\n" + "="*60)
print(f"最終汎化テスト（{MAX_DIGITS}桁の未学習データのみで評価）")
print("="*60)

final_acc = evaluate_generalization(
    final_holdout_enc, final_holdout_dec, final_holdout_tgt, final_holdout_q_raws,
    stage_label=f"最終（{MAX_DIGITS}桁・未学習データ）",
    n_show=10, n_eval_max=1000
)

# ============================================================
# 9. まとめ表示
# ============================================================
print("\n" + "="*60)
print("汎化スコアまとめ（ステージごと）")
print("="*60)
for digits, acc in generalization_scores:
    bar = "█" * int(acc / 5)
    print(f"  Stage {digits} ({digits}桁): {acc:5.1f}% {bar}")
print(f"\n  最終（{MAX_DIGITS}桁・一度も学習していないデータ）: {final_acc:.1f}%")
print("="*60)
