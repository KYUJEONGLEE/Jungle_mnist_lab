# -*- coding: utf-8 -*-
"""손실 함수 모음."""

import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross Entropy Error (배치 평균).
    y_pred: (batch_size, 10) 확률
    단위 테스트에서는 (batch_size = 2, num_class = 3) 이다.
    이 말은 두 개의 데이터가 존재하고, 각 정답 클래스는 3개 라는 뜻
    y_true: (batch_size,) 정수 레이블 0~9
    정답 레이블을 표현하는 방식 2가지
    1. 클래스 인덱스 방식
        np.array([2,0])
    2. 원-핫 벡터 방식
        np.array([
            [0, 0, 1],
            [1, 0, 0]
        ])
    두 개의 방식 다 같은 뜻이다.
    """
    # TODO: 정답 클래스 확률의 log 값을 이용해 batch 평균 cross entropy를 계산하세요.
    # 힌트: np.clip으로 log(0)을 피하고, np.arange(batch_size)로 정답 위치를 고릅니다.
    eps = 1e-7
    batch_size = y_pred.shape[0]

    pred = y_pred[np.arange(batch_size), y_true]
    """
        각 데이터마다, 그 데이터의 정답 클래스에 해당하는 예측 확률만 뽑아라.
        클래스 인덱스 방식이라 곱하기 필요없이 바로 그 확률에 해당하는 pred를 뽑아오면 된다.

        문법이 조금 헷갈림
        반복문으로
        pred = []

        for i in range(batch_size):
            pred.append(y_pred[i, y_true[i]])
    """
    pred = np.clip(pred, eps, 1)

    loss = -np.sum(np.log(pred)) / batch_size

    return loss
