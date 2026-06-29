"""
lstm_add_seq2seq.py  ― カリキュラム学習版 Seq2Seq 加算モデル

【汎化できなかった原因】
  「3桁で学習 → 5桁でテスト」は分布外汎化の問題です。
  LSTM は足し算のアルゴリズムを学ぶのではなく、
  学習時に見た長さのパターンを記憶します。
  学習時と全く異なる入力長を与えると予測が崩壊します。

【解決策：カリキュラム学習 (Curriculum Learning)】
  1桁 → 2桁 → … → MAX_DIGITS桁 と、段階的に難易度を上げながら学習する。
  各ステージで前のステージのデータも混ぜて忘却を防ぐ（リプレイ）。
  これにより「足し算のルール」を般化的に習得させる。

【テスト】
  TEST_DIGITS を MAX_DIGITS+1 〜 +2 程度に設定することで、
  学習で見たことのない桁数への汎化性能を評価できる。
  （あまり大きく外れると LLM でも難しいため +1〜2 が現実的）
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# ハイパーパラメータ
# ============================================================
MAX_DIGITS    = 4     # カリキュラムの最大桁数（ここまで段階的に学習）
TEST_DIGITS   = 5     # テスト桁数（MAX_DIGITS+1 が目安）
REVERSE       = True  # 入力を逆順にする（右から読むと桁上がりが先に来て学習しやすい）
LATENT_DIM    = 256   # LSTM 隠れユニット数（汎化のため少し増やす）

# カリキュラム学習の各ステージ設定
STAGE_SAMPLES = 20000  # 各ステージのサンプル数
STAGE_EPOCHS  = 20     # 各ステージのエポック数
BATCH_SIZE    = 128
REPLAY_RATIO  = 0.3    # 過去ステージのデータを混ぜる割合

TEST_SAMPLES  = 500    # テストサンプル数

# ============================================================
# 文字テーブル
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
        for i, c in enumerate(s):
            x[i, self.char_indices[c]] = 1.0
        return x

    def decode(self, x: np.ndarray, calc_argmax: bool = True) -> str:
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[int(i)] for i in x)


ctable   = CharacterTable(CHARS)
NUM_CHARS = ctable.num_chars

# ============================================================
# 定数：最大入力長・最大出力長（全ステージ共通の固定長パディング）
# ============================================================
# すべての入力を MAX_DIGITS に合わせてパディングすることで
# テンソル形状を固定し、バッチ処理を可能にする
GLOBAL_MAXLEN  = MAX_DIGITS + 1 + MAX_DIGITS  # 例: 4+1+4=9
GLOBAL_ANS_LEN = MAX_DIGITS + 2               # 例: 4+2=6  (最大桁上がり考慮)

# テスト用の長さ（TEST_DIGITS に対応した長さ）
TEST_MAXLEN  = TEST_DIGITS + 1 + TEST_DIGITS
TEST_ANS_LEN = TEST_DIGITS + 2


# ============================================================
# データ生成関数
# ============================================================
def generate_data(num_samples: int, digits: int, reverse: bool = True,
                  enc_maxlen: int = None, ans_maxlen: int = None):
    """
    指定桁数の加算データを生成し one-hot テンソルに変換して返す。

    enc_maxlen / ans_maxlen を指定すると、その長さでパディングする。
    省略時は digits に基づいて自動決定。
    """
    if enc_maxlen is None:
        enc_maxlen = digits + 1 + digits
    if ans_maxlen is None:
        ans_maxlen = digits + 2

    def rand_number():
        n = np.random.randint(1, digits + 1)
        return int("".join(np.random.choice(list("0123456789")) for _ in range(n)))

    questions, answers = [], []
    seen = set()
    while len(questions) < num_samples:
        a, b = rand_number(), rand_number()
        key  = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q   = f"{a}+{b}"
        q   = q   + " " * (enc_maxlen - len(q))
        ans = str(a + b)
        ans = ans + " " * (ans_maxlen - len(ans))
        if reverse:
            q = q[::-1]
        questions.append(q)
        answers.append(ans)

    N = len(questions)
    enc_in  = np.zeros((N, enc_maxlen, NUM_CHARS), dtype=np.float32)
    dec_in  = np.zeros((N, ans_maxlen, NUM_CHARS), dtype=np.float32)
    dec_tgt = np.zeros((N, ans_maxlen, NUM_CHARS), dtype=np.float32)

    for i, (q, ans) in enumerate(zip(questions, answers)):
        enc_in[i]  = ctable.encode(q,   enc_maxlen)
        dec_tgt[i] = ctable.encode(ans, ans_maxlen)
        # Teacher-Forcing: BOS(スペース) + ans[:-1]
        bos = np.zeros(NUM_CHARS, dtype=np.float32)
        bos[ctable.pad_idx] = 1.0
        dec_in[i, 0]  = bos
        dec_in[i, 1:] = ctable.encode(ans, ans_maxlen)[:-1]

    # シャッフル
    idx = np.random.permutation(N)
    return enc_in[idx], dec_in[idx], dec_tgt[idx], questions, answers


# ============================================================
# モデル構築（入力長を None で可変にする）
# ============================================================
print("モデル構築中...")

# Encoder
encoder_inputs = keras.Input(shape=(None, NUM_CHARS), name="encoder_input")
encoder_lstm   = layers.LSTM(LATENT_DIM, return_state=True, name="encoder_lstm")
_, enc_h, enc_c = encoder_lstm(encoder_inputs)
encoder_states  = [enc_h, enc_c]

# Decoder
decoder_inputs  = keras.Input(shape=(None, NUM_CHARS), name="decoder_input")
decoder_lstm    = layers.LSTM(LATENT_DIM, return_sequences=True,
                               return_state=True, name="decoder_lstm")
decoder_out, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense   = layers.Dense(NUM_CHARS, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_out)

model = keras.Model(
    inputs  = [encoder_inputs, decoder_inputs],
    outputs = decoder_outputs,
    name    = "seq2seq_train"
)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


# ============================================================
# 推論モデル
# ============================================================
inf_encoder = keras.Model(
    inputs  = encoder_inputs,
    outputs = encoder_states,
    name    = "inf_encoder"
)

inf_dec_input = keras.Input(shape=(1, NUM_CHARS), name="inf_dec_input")
inf_state_h   = keras.Input(shape=(LATENT_DIM,),  name="inf_state_h")
inf_state_c   = keras.Input(shape=(LATENT_DIM,),  name="inf_state_c")
inf_dec_out, inf_h, inf_c = decoder_lstm(
    inf_dec_input, initial_state=[inf_state_h, inf_state_c]
)
inf_dec_out = decoder_dense(inf_dec_out)
inf_decoder = keras.Model(
    inputs  = [inf_dec_input, inf_state_h, inf_state_c],
    outputs = [inf_dec_out, inf_h, inf_c],
    name    = "inf_decoder"
)


def decode_sequence(enc_input_seq: np.ndarray, ans_len: int) -> str:
    """自己回帰デコード。enc_input_seq: shape (1, seq_len, NUM_CHARS)"""
    h, c = inf_encoder.predict(enc_input_seq, verbose=0)

    target_seq = np.zeros((1, 1, NUM_CHARS), dtype=np.float32)
    target_seq[0, 0, ctable.pad_idx] = 1.0

    result = []
    for _ in range(ans_len):
        out, h, c = inf_decoder.predict([target_seq, h, c], verbose=0)
        idx = np.argmax(out[0, 0])
        result.append(ctable.indices_char[idx])
        target_seq = np.zeros((1, 1, NUM_CHARS), dtype=np.float32)
        target_seq[0, 0, idx] = 1.0

    return "".join(result)


# ============================================================
# カリキュラム学習
# ============================================================
all_histories = []
replay_enc, replay_dec_in, replay_dec_tgt = None, None, None  # リプレイバッファ

print("\n" + "="*60)
print("カリキュラム学習開始")
print("="*60)

for stage_digits in range(1, MAX_DIGITS + 1):
    print(f"\n--- Stage {stage_digits}: {stage_digits}桁 ---")

    # 現ステージのデータ生成（GLOBAL_MAXLEN/ANS_LEN にパディング）
    enc_x, dec_x, dec_y, _, _ = generate_data(
        STAGE_SAMPLES, stage_digits, REVERSE,
        enc_maxlen=GLOBAL_MAXLEN, ans_maxlen=GLOBAL_ANS_LEN
    )

    # リプレイデータを混ぜる（過去ステージの忘却防止）
    if replay_enc is not None:
        n_replay = int(len(enc_x) * REPLAY_RATIO)
        idx_r    = np.random.choice(len(replay_enc), n_replay, replace=False)
        enc_x    = np.concatenate([enc_x,    replay_enc[idx_r]],    axis=0)
        dec_x    = np.concatenate([dec_x,    replay_dec_in[idx_r]], axis=0)
        dec_y    = np.concatenate([dec_y,    replay_dec_tgt[idx_r]],axis=0)
        # シャッフル
        perm  = np.random.permutation(len(enc_x))
        enc_x, dec_x, dec_y = enc_x[perm], dec_x[perm], dec_y[perm]

    # リプレイバッファを更新（現ステージのデータを追加）
    if replay_enc is None:
        replay_enc    = enc_x[:STAGE_SAMPLES]
        replay_dec_in  = dec_x[:STAGE_SAMPLES]
        replay_dec_tgt = dec_y[:STAGE_SAMPLES]
    else:
        replay_enc    = np.concatenate([replay_enc,    enc_x[:STAGE_SAMPLES]])
        replay_dec_in  = np.concatenate([replay_dec_in,  dec_x[:STAGE_SAMPLES]])
        replay_dec_tgt = np.concatenate([replay_dec_tgt, dec_y[:STAGE_SAMPLES]])

    split = len(enc_x) - len(enc_x) // 10
    hist = model.fit(
        [enc_x[:split], dec_x[:split]], dec_y[:split],
        batch_size      = BATCH_SIZE,
        epochs          = STAGE_EPOCHS,
        validation_data = ([enc_x[split:], dec_x[split:]], dec_y[split:]),
        verbose         = 1,
    )
    all_histories.append((stage_digits, hist))

    # ステージ終了時に簡易評価（現ステージの検証データ 5 件表示）
    print(f"\n  [ステージ{stage_digits} サンプル確認]")
    for i in range(5):
        q_raw   = enc_x[split + i]   # shape: (GLOBAL_MAXLEN, NUM_CHARS)
        ans_raw = dec_y[split + i]
        inp     = q_raw[np.newaxis]
        pred    = decode_sequence(inp, GLOBAL_ANS_LEN).strip()
        correct = ctable.decode(ans_raw).strip()
        q_disp  = ctable.decode(q_raw)
        q_disp  = q_disp[::-1] if REVERSE else q_disp
        mark    = "〇" if pred == correct else "×"
        print(f"    Q: {q_disp.strip():>10}  正解: {correct:<6}  予測: {pred:<6}  {mark}")


# ============================================================
# テスト（TEST_DIGITS 桁：学習範囲外）
# ============================================================
print(f"\n{'='*60}")
print(f"テスト: TEST_DIGITS={TEST_DIGITS}  (学習最大桁数 {MAX_DIGITS} を超える桁数)")
print(f"{'='*60}")

enc_test, _, _, test_q_strs, test_ans_strs = generate_data(
    TEST_SAMPLES, TEST_DIGITS, REVERSE,
    enc_maxlen=TEST_MAXLEN, ans_maxlen=TEST_ANS_LEN
)

correct_count = 0
show_n        = 20
print(f"\n--- テスト結果 (最初の {show_n} 件) ---")

for i in range(len(enc_test)):
    inp     = enc_test[np.newaxis, i]  # (1, TEST_MAXLEN, NUM_CHARS)
    pred    = decode_sequence(inp, TEST_ANS_LEN).strip()
    correct = test_ans_strs[i].strip()
    q_disp  = test_q_strs[i][::-1].strip() if REVERSE else test_q_strs[i].strip()

    if pred == correct:
        correct_count += 1

    if i < show_n:
        mark = "〇" if pred == correct else "×"
        print(f"  Q: {q_disp:>14}  正解: {correct:<8}  予測: {pred:<8}  {mark}")

accuracy = correct_count / len(enc_test) * 100
print(f"\nテスト正解率: {correct_count}/{len(enc_test)} = {accuracy:.1f}%")


# ============================================================
# 学習曲線の描画（各ステージを色分け）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
colors = plt.cm.tab10.colors

offset = 0
for stage_digits, hist in all_histories:
    n  = len(hist.history["loss"])
    ep = range(offset + 1, offset + n + 1)
    c  = colors[(stage_digits - 1) % 10]
    axes[0].plot(ep, hist.history["loss"],     color=c, label=f"{stage_digits}桁 Train")
    axes[0].plot(ep, hist.history["val_loss"], color=c, linestyle="--")
    axes[1].plot(ep, hist.history["accuracy"],     color=c, label=f"{stage_digits}桁 Train")
    axes[1].plot(ep, hist.history["val_accuracy"], color=c, linestyle="--")
    # ステージ境界の縦線
    if offset > 0:
        axes[0].axvline(offset + 0.5, color="gray", linewidth=0.8, linestyle=":")
        axes[1].axvline(offset + 0.5, color="gray", linewidth=0.8, linestyle=":")
    offset += n

for ax, title, ylabel in zip(axes,
                              ["Loss (実線=Train / 破線=Val)", "Accuracy"],
                              ["Loss", "Accuracy"]):
    ax.set_title(title)
    ax.set_xlabel("Epoch (累積)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("seq2seq_curriculum_history.jpg", dpi=120)
plt.show()
print("学習曲線を seq2seq_curriculum_history.jpg に保存しました。")
