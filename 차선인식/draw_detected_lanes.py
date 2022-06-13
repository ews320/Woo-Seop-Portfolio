import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """도로 이미지를 가져와서 모델에 맞게 크기를 조정하고,
       모델에서 그릴 차선을 G 색상으로 예측하고, 차선의
       RGB 이미지를 재생성하고 원래 도로 이미지.
    """

    # 모델에 공급할 이미지 준비
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # 신경망으로 예측(255를 곱하여 값 비정규화)
    prediction = model.predict(small_img)[0] * 255

    # 평균화를 위해 목록에 차선 예측 추가
    lanes.recent_fit.append(prediction)
    # 평균을 위해 마지막 5개만 사용
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # 평균 감지 계산
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # 가짜 R & B 색상 치수 생성, G와 스택
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # 원본 이미지에 맞게 크기 조정
    lane_image = imresize(lane_drawn,
                          (1080, 1920, 3))  # (720, 1280, 3)

    # 레인 도면을 원본 이미지에 병합
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


if __name__ == '__main__':
    # Keras 모델 로드
    model = load_model('full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()

    # 출력 비디오를 저장할 위치
    vid_output = 'city_day2_out2.mp4'
    # 입력 영상의 위치
    clip1 = VideoFileClip("city_day2.mp4")
    # 클립 만들기
    vid_clip = clip1.fl_image(road_lines)
    vid_clip.write_videofile(vid_output, audio=False)
