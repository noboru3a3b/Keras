import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# train set size
TRAINING_SIZE = 50000
# 最大桁数
DIGITS = 3
# Reverseの有無
REVERSE = True

# ベクトル化するためのクラス
class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i,c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i,c in enumerate(self.chars))
        
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)

# 入力される文字列の最大長
MAXLEN = DIGITS + 1 + DIGITS

chars = "0123456789+ "
ctable = CharacterTable(chars) # インスタンス化

questions = []
expected = []
seen = set()
print("Generating data...")
while len(questions)<TRAINING_SIZE: # train size回反復
    f = lambda: int(
    "".join( # 空白文字列に足していく
    np.random.choice(list("0123456789")) # 1桁の数字を1文字選ぶ
    for i in range(np.random.randint(1,DIGITS+1)) # 1～桁数回繰り返す
    )
    )
    
    a,b = f(),f() # a,b 2つの数字を無名関数fから取得
    key = tuple(sorted((a,b)))
    if key in seen: # 2数の組み合わせが既にseenに定義されているとき
        continue
    seen.add(key)
    q = "{}+{}".format(a,b) # a+b文字列に変換
    query = q + " " * (MAXLEN - len(q)) # パディング
    ans = str(a + b) # 正解文字列
    ans +=" " *(DIGITS+1-len(ans)) # パディング
    if REVERSE: # Reverse処理
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print("Total questions:",len(questions))
#print(questions)
#print(questions[:5],len(questions))
#print(expected[:5],len(expected))

print("Vectorization...")
x = np.zeros((len(questions),MAXLEN,len(chars)),dtype=np.bool_)
y = np.zeros((len(questions),DIGITS+1,len(chars)),dtype=np.bool_)

for i,sentence in enumerate(questions):
    x[i] = ctable.encode(sentence,MAXLEN)
for i,sentence in enumerate(expected):
    y[i] = ctable.encode(sentence,DIGITS+1)
    
indices = np.arange(len(y))
np.random.shuffle(indices)
x=x[indices]
y=y[indices]

split_at = len(x) -len(x)//10
(x_train,x_val) = x[:split_at],x[split_at:]
(y_train,y_val) = y[:split_at],y[split_at:]

print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)

#print(x_train[0])
#print(y_train[0])

# モデル定義
print("Build model...")
num_layers = 1

model = keras.Sequential()
model.add(layers.LSTM(128,input_shape=(MAXLEN,len(chars))))
model.add(layers.RepeatVector(DIGITS+1))
for _ in range(num_layers):
    model.add(layers.LSTM(128,return_sequences=True))
    
model.add(layers.Dense(len(chars),activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
print("complete")

# 学習
"""
# （Reverseなしの確認用）
# 学習の途中でvalidationを表示する
epochs = 30
batch_size = 12

for epoch in range(1,epochs): # 1epoch毎にvalからランダムに抜き出して精度を確認
    history = model.fit(x_train,y_train,batch_size=batch_size,epochs=1,
                       validation_data=(x_val,y_val),)

    for i in range(10):
        ind = np.random.randint(0,len(x_val))
        rowx,rowy = x_val[np.array([ind])],y_val[np.array([ind])]
        preds = np.argmax(model.predict(rowx),axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0],calc_argmax=False)
        print("Q", q[::-1] if REVERSE else q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            print("〇 " + guess)
        else:
            print("× " + guess)
"""
# Reverseありの場合
# 学習の途中でvalidationを表示しない
epochs = 30
batch_size = 12

history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,
                    validation_data=(x_val,y_val),)

def plot_loss(history):
    """エポックごとの損失関数をプロットする関数
    
    Args:
    history : fittingの履歴
    
    Returns:
    None
    """
    # 損失関数の履歴を取得
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]

    # 損失関数をプロット
    epochs=range(1,len(history.history["loss"])+1)
    plt.figure(facecolor="white")
    plt.plot(epochs,loss_train,label="Training loss")
    plt.plot(epochs,loss_val,label="Validation loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig("loss.jpg")
    plt.show()
    
def plot_acc(history):
    """エポックごとの正解率をプロットする関数
    
    Args:
    history : fittingの履歴
    
    Returns:
    None
    """
    acc_train = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    epochs = range(1,len(history.history["accuracy"])+1)
    plt.figure(facecolor="white")
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='Validation accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig("acc.jpg")
    plt.show()

# 損失関数の描画
plot_loss(history)

# 正解率を描画
plot_acc(history)

