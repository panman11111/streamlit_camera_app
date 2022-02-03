import streamlit as st
import cv2
import time
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

st.markdown("# Camera Application ")
cascade_path = 'haarcascade_frontalface_alt.xml'

device = user_input = st.text_input("input your video/camera device ", "0")
st.markdown("Please wait for about 1 minute")
if device.isnumeric():
    device = int(device)

# カメラを取得
cap = cv2.VideoCapture(device)

# カメラの画像を取得
ret, img = cap.read()

# 実行開始時刻を秒単位で保存
change_moment = time.time()

# 動画表示欄を作成
image_loc = st.empty()

# グラフ描写エリアを削除
plot_area = st.empty()

# グラフ作成の設定
fig = plt.figure()
ax = fig.add_subplot()
plot_area.pyplot(fig)
rows = np.array([0])

Flag = True
while cap.isOpened:
    # リストの長さを取得
    rows_length = len(rows)
    # カメラの値を取得
    ret, img = cap.read()
    # グレースケールに変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 学習済モデルの取得
    cascade = cv2.CascadeClassifier(cascade_path)
    
    face_list = cascade.detectMultiScale(img_gray, minSize = (20, 20))
    count = len(face_list)

    if abs(int(change_moment - time.time())) > 1:
        change_moment = time.time()
        ax.clear()
        # データを追加する
        rows = np.append(rows, count)
        # グラフを描画し直す
        ax.grid()
        if Flag:
            pass
        else:
            ax.set_xlim(rows_length-20, rows_length)

        ax.set_xticks([i for i in range(rows_length+ 1)][-20:])
        ax.plot(rows)
        # プレースホルダに書き出す
        plot_area.pyplot(fig)
    # 見つかった顔を順番に囲う
    for (x, y, w, h)in face_list:
        # 四角を描写
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), thickness=3)
        # 画面右上にカウントを表示
        cv2.putText(img, 
        f"count:{count}", 
        org=(415, 50), 
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.5,
        color=(0, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA)

    # グラフの表示制御
    if rows_length > 20 & Flag:
        Flag = False
    # 表示画像の更新
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_loc.image(img)

cap.release()
