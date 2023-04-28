import pyaudio
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
from model import Wav2Vec2ForCTCnCLS

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1].copy()
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1].copy()
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def cv2_putText(img, text, org, fontFace, fontScale, color=(255, 0, 0)):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    w, h = draw.textsize(text, font = fontPIL)
    draw.text(xy = (x,y-h), text = text, fill = colorRGB, font = fontPIL)
    imgCV = pil2cv(imgPIL)
    return imgCV

SAMPLE_RATE = 16000             # サンプリングレート
FRAME_SIZE = 512               # フレームサイズ
INT16_MAX = 32767               # サンプリングデータ正規化用
SAMPLING_SIZE = FRAME_SIZE * 4  # サンプリング配列サイズ
AI_SAMPLING_SIZE = 16384        # AIに入力するサンプルサイズ
AI_FREQ = 8
RADIUS = 120
WIDTH = 800     # 表示領域の幅
HEIGHT = 600    # 表示領域の高さ
FONT = "Fonts/ヒラギノ明朝 ProN.ttc" # 日本語文字のフォント
TEXT_SIZE = 120 # 日本語文字の大きさ
MODEL_PATH = "AI/ckpts/01M" # チェックポイントのパス

model = Wav2Vec2ForCTCnCLS.from_pretrained(MODEL_PATH)
emotion_map_j = {0 : '平', 1 : '喜', 2 : '怒', 3 : '悲'}
emotion_map_e = {0 : 'Neutral', 1 : 'Happy', 2 : 'Anger',  3 : 'Sad'}
emotion_map_color = {0 : (34, 139, 34), 1 : (0, 130, 255), 2 : (0, 0, 255), 3: (255, 0, 0)}

# 周波数成分を表示用配列に変換する用の行列(spectram_array)作成
#   FFT結果（周波数成分の配列)から、どの要素を合計するかをまとめた行列
spectram_range = [int(22050 / 2 ** (i/10)) for i in range(100, -1,-1)]    # 21Hz～22,050Hzの間を分割
freq = np.abs(np.fft.fftfreq(SAMPLING_SIZE, d=(1/SAMPLE_RATE)))  # サンプル周波数を取得
spectram_array = (freq <= spectram_range[0]).reshape(1,-1)
for index in range(1, len(spectram_range)):
    tmp_freq = ((freq > spectram_range[index - 1]) & (freq <= spectram_range[index])).reshape(1,-1)
    spectram_array = np.append(spectram_array, tmp_freq, axis=0)

# 表示用の変数定義・初期化
part_w = WIDTH / len(spectram_range)
part_h = HEIGHT / 100
img = np.full((HEIGHT, WIDTH, 3), 0, dtype=np.uint8)

# マイク サンプリング開始
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                    input=True, input_device_index=0, frames_per_buffer=FRAME_SIZE)

# サンプリング配列(sampling_data)の初期化
sampling_data = np.zeros(SAMPLING_SIZE)
ai_sampling_data = np.zeros(AI_SAMPLING_SIZE)

# count および pred_id の初期化
count = 0
pred_id = 0

# グラデーションの生成
def part_gradation(start, end, num):
    grad_b = np.linspace(start[2], end[2], num)
    grad_g = np.linspace(start[1], end[1], num)
    grad_r = np.linspace(start[0], end[0], num)
    
    return [(grad_b[i], grad_g[i], grad_r[i]) for i in range(num)]

def gradation(length):
    half_target_colors = {(255, 97, 97) : 0, (233, 178, 45) : 20, (192, 202, 75): 34, (53, 179, 56) : 58, (86, 110, 243) : 79, (154, 39, 238) : 100}
    colors = list(half_target_colors.keys())
    sep_points = list(half_target_colors.values())
    between = np.array([sep_points[i] - sep_points[i - 1] for i in range(1, len(sep_points))]) * length / 200
    
    grad = []
    for i in range(0, len(between)):
        grad.extend(part_gradation(start=colors[i], end=colors[i + 1], num=int(between[i])))
    
    grad.extend(reversed(grad))

    if (len(grad)) > length:
        grad = grad[:length]
    elif(len(grad)) < length:
        l = len(grad)
        for i in range(length - l):
            grad.append(grad[-1])
    
    return grad

grad = gradation(length=spectram_array.shape[0])

while True:
    cv2.rectangle(img, (0,0), (WIDTH, HEIGHT), (255,255,255), thickness=-2)   # 出力領域のクリア

    # フレームサイズ分データを読み込み
    frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
    # サンプリング配列に読み込んだデータを追加
    frame_data = np.frombuffer(frame, dtype="int16") / INT16_MAX
    sampling_data = np.concatenate([sampling_data, frame_data])
    ai_sampling_data = np.concatenate([ai_sampling_data, frame_data])
    if sampling_data.shape[0] > SAMPLING_SIZE:
        # サンプリング配列サイズよりあふれた部分をカット
        sampling_data = sampling_data[sampling_data.shape[0] - SAMPLING_SIZE:]
    if ai_sampling_data.shape[0] > AI_SAMPLING_SIZE:
        # 同様に
        ai_sampling_data = ai_sampling_data[ai_sampling_data.shape[0] - AI_SAMPLING_SIZE:]

    # 高速フーリエ変換（周波数成分に変換）
    fft = np.abs(np.fft.fft(sampling_data))

    # 表示用データ配列作成
    #   周波数成分の値を周波数を範囲毎に合計して、表示用データ配列(spectram_data)を作成
    spectram_data = np.dot(spectram_array, fft)

    # AIにより感情を推定
    if (count % AI_FREQ == 0):
        with torch.no_grad():
            tensor = torch.tensor(ai_sampling_data, dtype=torch.float32).reshape(1, AI_SAMPLING_SIZE)
            output = model(tensor, if_ctc=False)
        logits = output['logits'][1]
        pred_id = int(np.argmax(logits, axis=-1))
        count = 0

     # 出力処理
    img = cv2_putText(img, text=emotion_map_j[pred_id], org=((WIDTH - TEXT_SIZE) / 2, (HEIGHT + TEXT_SIZE) / 2), 
                      color=emotion_map_color[pred_id], fontFace=FONT, fontScale=TEXT_SIZE)
    for index, value in enumerate(spectram_data):
        # 単色のグラフとして表示
        rad = (2 * np.pi) * (index / len(spectram_data))
        x1 = int(WIDTH / 2 + np.sin(rad) * RADIUS)
        y1 = int(HEIGHT / 2 - np.cos(rad) * RADIUS)
        rad = (2 * np.pi) * (index / len(spectram_data))
        x2 = int(WIDTH / 2 + np.sin(rad) * (RADIUS + value/2))
        y2 = int(HEIGHT / 2 - np.cos(rad) * (RADIUS + value/2))
        cv2.line(img, (x1, y1), (x2, y2), grad[index], thickness=2)
        
    # 画面表示
    # cv2.putText(img, text=emotion_map_e[pred_id], org=(100, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=(255, 0, 0))
    cv2.imshow("Emo Vis", img)

    # 終了キーチェック
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 0x1b:
        break

    count += 1

# マイク サンプリング終了処理
stream.stop_stream()
stream.close()
audio.terminate()