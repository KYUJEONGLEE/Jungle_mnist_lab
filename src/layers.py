# -*- coding: utf-8 -*-
"""
신경망 layer 모음.

학생 구현 대상:
- Affine.forward, Affine.backward
- BatchNorm.forward, BatchNorm.backward
- Dropout.forward, Dropout.backward
"""

import numpy as np


class Affine:
    """
    완전연결층(Fully Connected Layer).

    수식은 y = xW + b 입니다.
    MNIST에서는 784개 픽셀 입력을 은닉층/출력층 차원으로 선형 변환하는 역할을 합니다.
    """

    def __init__(self, W, b):
        """가중치 W와 편향 b를 외부 params dict와 같은 배열 객체로 공유합니다."""
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)

        Returns:
            (batch_size, output_dim)
        """
        # TODO: backward에서 사용할 입력 x를 저장하고 x @ W + b를 반환하세요.
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        """
        Args:
            dout: (batch_size, output_dim)
            Affine 층의 출력 y가 최종 Loss에 얼마나 영향을 주는가?
            즉 dL/dy, 뒤쪽 층에서 넘어온 기울기

        Returns:
            dx: (batch_size, input_dim)
            dx -> 앞 층으로 넘길 기울기, 입력 x가 Loss에 얼마나 영향을 줬는가?

        Side effects:
            self.dW, self.db에 optimizer가 사용할 gradient를 저장합니다.
            dW : W를 얼마나 바꿔야 하는가
            db : b를 얼마나 바꿔야 하는가
        """
        # TODO: self.dW, self.db, dx를 계산하세요.
        # 힌트: dW = x.T @ dout, db = batch 방향 합, dx = dout @ W.T
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class BatchNorm:
    """
    Batch Normalization.

    미니배치 단위로 각 feature의 평균과 분산을 맞춰 학습을 안정화합니다.
    train=True일 때는 현재 배치 통계를 쓰고, 추론 때는 누적 running_mean/running_var를 사용합니다.
    """

    def __init__(self, gamma, beta, momentum=0.9):
        """
        Args:
            gamma: 정규화된 값을 다시 scale하는 학습 파라미터
            beta: 정규화된 값에 더하는 shift 학습 파라미터
            momentum: running_mean/running_var 이동평균 비율
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = np.zeros_like(beta)
        self.running_var = np.zeros_like(beta)
        self.eps = 1e-7

    def forward(self, x, train=True):
        """
        Args:
            x: (batch_size, feature_dim)
            train: True면 배치 통계, False면 running 통계 사용

        Returns:
            정규화 후 gamma, beta가 적용된 배열
        """
        # TODO: train=True에서는 batch mean/var로 정규화하고 running 통계를 갱신하세요.
        # TODO: train=False에서는 running_mean/running_var를 사용하세요.
        """
        1. 일단 x의 평균과 분산을 구한다.
        공식은 교재 p.211 참고
        주의 해야 할 점)np.mean()을 사용할 때 axis = 0을 설정해줘야 함.
                    배치 정규화의 핵심은 "이 배치 안에서 같은 위치의 값들끼리의 평균,분산"
                    을 요구하는것이지, 한 이미지의 모든 픽셀값에 대한 평균,분산을 요구하는게 아님
        """
        if train:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            """ 2. 구한 평균과 분산이 1이 되도록 하는 정규화 식"""
            x_centered = x - mean
            std = np.sqrt(var + self.eps)
            x_hat = x_centered / std

            """
            3. running_mean, var 갱신
                위 개념을 쓰는 이유부터 알아야 한다.
                학습하는 동안은 위 개념이 필요없다.
                하지만 테스트를 돌릴때 필요한 경우가 생긴다.

                만약 테스트 배치의 크기가 1개 라고 해보자.
                위에서 구한 mean이 x 그 자체가 되버리는 경우가 생기는데
                즉 x - mean = 0 이 될 수도 있다.

                따라서 테스트 때는 현재 배치의 평균을 사용하는게 아니라,
                학습하는 동안 봤던 평균들을 누적해서 만든 대표 평균을 사용
                그게 running_mean : 기존 누적 평균의 90% 반영 + 이번 배치의 평균을 10% 반영
            """
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var

            self.x_centered = x_centered
            self.std = std
            self.x_hat = x_hat
            self.batch_size = x.shape[0]
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * x_hat + self.beta
        return out

    def backward(self, dout):
        """
        BatchNorm 입력 x, scale gamma, shift beta에 대한 gradient를 계산합니다.

        Args:
            dout: 다음 층에서 넘어온 gradient

        Returns:
            dx: BatchNorm 입력 x에 대한 gradient
        """
        # TODO: self.dbeta, self.dgamma, dx를 계산하세요.
        # 힌트: 먼저 dbeta와 dgamma shape가 beta/gamma와 같은지 확인합니다.

        """
            1. dbeta의 경우
            out = gamma * x_hat + beta
            beta는 BatchNorm 출력에 그냥 더해지는 값이다.

            out = gamma * x_hat + beta 이므로,
            out을 beta에 대해 미분하면 1이다.

            따라서 뒤쪽 층에서 흘러온 gradient dout에 1을 곱한 값이 beta 쪽 gradient가 된다.

            그리고 beta는 feature마다 하나씩 있고, 같은 feature의 모든 배치 데이터에 공통으로 더해지므로,
            배치 방향(axis=0)으로 dout을 모두 더하면 beta의 gradient가 된다.
        """
        self.dbeta = np.sum(dout, axis=0)
        """
            2. dgamma의 경우
            out = gamma * x_hat + beta

            위 식을 gamma에 대해서 미분하면 x_hat이 나온다.
            beta와 마찬가지로 흘러온 gradient에 x_hat을 곱한 값이 gradient가 된다.

            아 x_hat을 저장
        """
        self.dgamma = np.sum(dout * self.x_hat, axis=0)
        """
            3. dx를 구하려면 dx_hat부터 구해야 한다
        """
        dx_hat = dout * self.gamma
        

class Dropout:
    """
    Dropout.

    학습 중 일부 뉴런 출력을 무작위로 0으로 만들어 과적합을 줄입니다.
    이 구현은 추론 시 출력에 (1 - drop_ratio)를 곱하는 기본 dropout 방식을 사용합니다.
    """

    def __init__(self, drop_ratio=0.5):
        """Args: drop_ratio: 학습 중 0으로 만들 뉴런 비율."""
        self.drop_ratio = drop_ratio

    def forward(self, x, train=True):
        """
        Args:
            x: 입력 배열
            train: True면 무작위 mask 적용, False면 평균적인 출력 크기로 scale
        """
        # TODO: train=True에서는 mask를 만들고 x에 곱하세요.
        # TODO: train=False에서는 x * (1 - drop_ratio)를 반환하세요.
        raise NotImplementedError("Dropout.forward를 구현하세요.")

    def backward(self, dout):
        """forward에서 꺼졌던 뉴런 위치에는 gradient도 흘리지 않습니다."""
        # TODO: forward에서 만든 mask를 dout에 곱하세요.
        raise NotImplementedError("Dropout.backward를 구현하세요.")
