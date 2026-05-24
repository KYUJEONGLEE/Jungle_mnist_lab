# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    y_true: (batch_size,) 정수 레이블 0~9
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    if y_pred.ndim == 1:  # ndim: 몇 차원 배열인지 반환
        y_true = y_true.reshape(1, y_true.size)  # 정답 배열 2차원으로 바꾸기, 데이터 1개짜리 입력도 배치 처리 코드와 똑같이 다루기 위함
        y_pred = y_pred.reshape(1, y_pred.size)

    batch_size = y_pred.shape[0]
    return -np.sum(np.log(y_pred[np.arange(batch_size), y_true] + 1e-7)) / batch_size