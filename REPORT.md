# MNIST 손글씨 인식 과제 보고서

## 0. 반·팀원


| 항목     | 내용            |
| ------ | ------------- |
| **반**  | SW_AI LAB 302반   |
| **팀원** | 송채강, 이규정, 이원재 |


---

## 1. 실험 목적

NumPy만을 사용해 MNIST 손글씨 숫자 분류 신경망을 직접 구현하고, 학습 과정을 통해 분류 성능을 확인하는 것을 목표로 한다. 또한 BatchNorm, Dropout, Adam optimizer를 적용했을 때 모델이 안정적으로 학습되는지 확인한다.

---

## 2. 모델 구조

MNIST 분류를 위한 완전연결 신경망(MLP) 모델이다. 입력 이미지는 28×28 픽셀을 펼친 784차원 벡터로 사용하며, 2개의 은닉층을 거쳐 10개 클래스 확률을 출력한다.

**전체 개요**
| 항목 | 내용 |
| --- | --- |
| 은닉층 수 | 2개 |
| 각 층 차원 | `784` → `512` → `256` → `10` |
| 활성화 함수 | 은닉층: `ReLU`, 출력층: `Softmax` |

**세부 구성**

<img width="1720" height="852" alt="image" src="https://github.com/user-attachments/assets/557de465-00c6-4b31-8345-da95a5383a2d" />


| 단계 | 구성 | 출력 차원 |
| --- | --- | --- |
| 입력층 | Flatten된 MNIST 이미지 | 784 |
| 은닉층 1 | `Affine` → `BatchNorm` → `ReLU` → `Dropout` | 512 |
| 은닉층 2 | `Affine` → `BatchNorm` → `ReLU` → `Dropout` | 256 |
| 출력층 | `Affine` → `Softmax` | 10 |

* 첫 번째 은닉층의 차원은 512, 두 번째 은닉층의 차원은 256이며, 두 은닉층 모두 활성화 함수로 `ReLU`를 사용한다. 
* 출력층은 10차원으로 구성되며, 최종 분류 확률 계산을 위해 `Softmax`를 적용한다. 
* 각 은닉층에서는 `Affine` 계층으로 선형 변환을 수행한 뒤 `BatchNorm`으로 활성값 분포를 정규화하고, 이후 `ReLU`와 `Dropout`을 순서대로 적용한다.

---

## 3. 학습 설정


| 항목                 | 값           |
| ------------------ | ----------- |
| 옵티마이저              | Adam        |
| 학습률 (lr)           | 0.001       |
| epochs             | 20          |
| batch_size         | 128         |
| Dropout 비율         | 0.5         |
| BatchNorm momentum | 0.9         |
| 가중치 초기화            | He (bias 0) |
| 학습 시간            | 3m ~ 3m 30s |

---

## 4. 정규화·규제

* 두 은닉층 모두에서 `BatchNorm`을 사용한다. 
* 각 `Affine` 계층 뒤에 `BatchNorm`을 배치해 활성값 분포를 정규화함으로써 학습이 더 안정적으로 진행되도록 했다. 
* `Dropout`도 두 은닉층 모두에 적용하며, 비율은 `0.5`로 설정되어 있다. 
* 학습 중 일부 뉴런을 무작위로 비활성화하여 과적합을 줄이도록 구성했다.

---

## 5. 초기화

* 가중치 초기화: `He` 초기화
* `np.random.randn(...) * sqrt(2 / 입력 차원)` 형태로 초기화하였다.
* bias는 0으로 설정하였다.
* 학습 초기에 기울기가 너무 작아지거나 커지는 문제를 줄이는 데 도움이 된다.

---

## 6. 결과


| 항목           | 값              |
| ------------ | -------------- |
| **테스트 정확도**  | 98.51%  |
| **총 파라미터 수** | 537,354 |


### 손실 커브

- 학습 곡선

<img width="722" height="660" alt="lr = 0 001 loss 그래프" src="https://github.com/user-attachments/assets/28203a7f-3ee4-4403-ac65-5214f175c618" />


---

## 7. 실험 분석
- Forward 과정에서는 각 layer를 순서대로 통과하며 예측 확률을 만들고, Backward 과정에서는 출력층에서 시작한 gradient가 역순으로 전달되며 각 파라미터의 gradient가 계산된다.

### 7.1 Drop-out Test
- BatchNorm은 각 은닉층의 입력 분포를 안정화하여 학습을 돕는 역할을 하고, `Dropout`은 일부 뉴런을 무작위로 비활성화하여 특정 뉴런에 과하게 의존하는 것을 줄인다.

- 두 설정 모두 비슷한 정확도를 보였고, 본 실행에서는 **Dropout ON**이 최종 정확도에서 근소하게 높았다.

- <img width="1019" height="558" alt="dropout on,off 정확도 그래프" src="https://github.com/user-attachments/assets/08da1b88-2292-4ab2-abcf-673d815ce095" />

---

- Dropout을 사용하지 않은 모델은 학습 손실이 더 빠르고 낮게 감소했다.
- 이는 Dropout을 끈 경우 모든 뉴런을 항상 사용하므로 학습 데이터를 더 쉽게 맞출 수 있기 때문이다.

- <img width="1019" height="557" alt="dropout on,off loss 그래프" src="https://github.com/user-attachments/assets/d199592c-1b94-4818-b380-774cb052db0c" />

</br>

- 반면 Dropout을 사용한 모델은 학습 손실은 더 높게 유지되었지만, 테스트 정확도는 비슷하거나 본 실행에서는 근소하게 더 높았다.
  ### 이는 Dropout이 학습을 어렵게 만들지만 `과적합`을 줄여 일반화 성능을 유지하는 역할을 할 수 있음을 보여준다.
---

### 7.2 Optimizer Test
- 본 실험에서는 나머지 조건은 동일하게 유지하고, optimizer만 `Adam`과 `SGD`로 변경하여 비교했다.
- `Adam`과 `SGD`는 업데이트 방식이 다르기 때문에 같은 learning rate가 같은 의미를 갖지 않는다. `SGD`는 gradient를 직접 사용해 파라미터를 갱신하므로 상대적으로 큰 learning rate가 필요할 수 있고, `Adam`은 gradient의 이동평균과 제곱 이동평균을 이용해 업데이트 크기를 조절하므로 더 작은 learning rate에서도 빠르게 수렴한다. 

  #### 따라서 본 실험에서는 `Adam=0.001`, `SGD=0.01`로 설정했다.

- <img width="1003" height="561" alt="Adam VS SGD acc 그래프" src="https://github.com/user-attachments/assets/03af2efd-d908-46c2-a19b-3493880a1704" />

- `Adam`은 학습 초반부터 정확도가 빠르게 상승했으며, 더 적은 epoch에서도 높은 정확도에 도달했다.
- 이는 `Adam`이 **gradient의 이동평균과 제곱 이동평균**을 이용해 **파라미터별 업데이트 크기를 조절**하기 때문이다.
- 반면 `SGD`는 단순히 현재 **gradient 방향으로만 파라미터를 갱신**하므로, Adam에 비해 초반 학습 속도가 느리게 나타났다.

- <img width="1009" height="558" alt="Adam VS SGD loss 그래프" src="https://github.com/user-attachments/assets/7d35ad6f-50af-4490-811b-4418563e5e62" />

- 손실 그래프에서도 `Adam`은 loss가 빠르게 감소하며 안정적으로 수렴하는 모습을 보였다.
- `SGD`는 loss가 감소하기는 했지만 Adam보다 완만하게 줄어들었고, 같은 epoch 내에서는 더 높은 loss를 유지했다.

---
### 7.3 Learning_rate Test

- <img width="1004" height="557" alt="epoch 50 lr accu 비교" src="https://github.com/user-attachments/assets/227bc2ab-08b0-4b68-8a11-873b9821e07b" />

- 정확도 그래프에서 `lr=0.001`은 초반부터 빠르게 높은 정확도에 도달했고, 이후에도 안정적으로 유지되었다.
- `lr=0.0001`은 learning rate가 작기 때문에 파라미터 업데이트 폭이 작아, 초반 학습 속도가 상대적으로 느리게 나타났다.
- 학습률이 작을 때, epoch를 20에서 50까지 증가시키고 테스트를 돌렸을 때 결과를 확인 한 결과 결국 **비슷한 값으로 수렴**하는 것을 볼 수 있었다.

- <img width="1007" height="557" alt="epoch 50 lr loss 비교" src="https://github.com/user-attachments/assets/27d11821-2771-463d-af57-5a07fdbca5da" />

- 손실 그래프에서도 `lr=0.001`은 안정적으로 loss가 감소하는 모습을 보였다.
- `lr=0.0001`은 loss가 감소하지만, 같은 epoch 안에서는 감소 속도가 느렸다.

- 추가적으로 lr = 1 일 때, 학습을 돌려 본 결과 어느 지점부터 학습에 실패했다.

 <img width="734" height="661" alt="lr = 1 loss 그래프" src="https://github.com/user-attachments/assets/e4fc019f-49cd-41b6-956a-f0fbb1081d9c" />

  ### 따라서 본 실험에서는 `Adam` optimizer의 learning rate로 `0.001`이 가장 안정적인 설정으로 나타났다.
