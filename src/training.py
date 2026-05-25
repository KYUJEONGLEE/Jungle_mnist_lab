# -*- coding: utf-8 -*-
"""학습 루프, 평가, 시각화 함수 모음."""

import matplotlib.pyplot as plt
import numpy as np

from losses import cross_entropy_loss


def train(model, optimizer, x_train, y_train, epochs=20, batch_size=128):
    """
    미니배치 학습 루프.

    한 배치마다 Forward -> Loss -> Backward -> Optimizer 업데이트 순서로 진행합니다.
    교육생은 이 함수에서 "예측값을 만들고, 손실을 계산하고, gradient로 파라미터를 바꾸는"
    전체 흐름을 확인할 수 있습니다.

    Returns:
        loss_history: epoch별 평균 손실 리스트
    """
    loss_history = []
    train_size = x_train.shape[0]

    # epoch는 전체 학습 데이터를 한 번씩 보는 단위
    for _ in range(epochs):
        # 매 epoch마다 데이터 순서를 섞어 모델이 순서에 익숙해지지 않게 설정한다.
        shuffled_idx = np.arange(train_size)
        np.random.shuffle(shuffled_idx)

        total_loss = 0.0
        batch_count = 0

        # 섞인 인덱스를 batch_size만큼 잘라 미니배치 생성
        for i in range(0, train_size, batch_size):
            batch_idx = shuffled_idx[i:i + batch_size]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Forward: 현재 파라미터로 예측값 계산
            y_pred = model.forward(x_batch, train=True)

            # Loss: 예측값과 정답의 차이를 숫자 하나로 계산
            loss = cross_entropy_loss(y_pred, y_batch)

            # Softmax + CrossEntropy를 합친 gradient
            # 정답 클래스 위치만 1을 빼고, 배치 크기로 나누어 평균 gradient로 만듬.
            dout = y_pred.copy()
            dout[np.arange(len(y_batch)), y_batch] -= 1
            dout /= len(y_batch)

            # Backward: 각 파라미터가 loss에 준 영향을 gradient로 계산
            grads = model.backward(dout)

            # BatchNorm의 gamma, beta gradient도 optimizer가 갱신할 수 있게 연결
            if getattr(model, "use_batchnorm", False):
                for name, layer in model.layers.items():
                    if name.startswith("BatchNorm"):
                        layer_num = name[-1]
                        grads[f"gamma{layer_num}"] = layer.dgamma
                        grads[f"beta{layer_num}"] = layer.dbeta

            # Update: optimizer가 gradient 방향을 보고 파라미터를 갱신
            optimizer.update(model.params, grads)

            total_loss += loss
            batch_count += 1

        # epoch 동안 나온 batch loss들의 평균을 기록
        loss_history.append(total_loss / batch_count)

    return loss_history


def evaluate(model, x, y):
    """정확도(%)와 총 파라미터 수 반환."""
    y_pred = model.predict(x)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == y) * 100
    total_params = sum(p.size for p in model.params.values())
    return accuracy, total_params


def plot_loss_history(loss_history):
    """손실 커브 그래프."""
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
