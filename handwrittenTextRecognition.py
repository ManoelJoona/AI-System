# 1. 必要なライブラリ
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 2. データの読み込みと前処理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. モデルの構築
model = Sequential([
    tf.keras.Input(shape=(28, 28)),      # 推奨されるInputの書き方
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. コンパイルと学習
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 5. テストデータで評価
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n🎯 テスト精度：{test_acc * 100:.2f}%")

# 6. 1枚目の画像＋予測表示
predictions = model.predict(x_test)
predicted_label = np.argmax(predictions[0])
true_label = np.argmax(y_test[0])

plt.figure(figsize=(3,3))
plt.imshow(x_test[0], cmap='gray')
plt.title(f"予測: {predicted_label}, 正解: {true_label}")
plt.axis('off')
plt.show()

# 7. ランダムに10枚を表示（予測＋正解）
random_indices = np.random.choice(len(x_test), size=10, replace=False)

plt.figure(figsize=(15, 4))
for i, idx in enumerate(random_indices):
    img = x_test[idx]
    true = np.argmax(y_test[idx])
    pred = np.argmax(predictions[idx])

    plt.subplot(1, 10, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"P:{pred}\nT:{true}")
    plt.axis('off')

plt.tight_layout()
plt.show()