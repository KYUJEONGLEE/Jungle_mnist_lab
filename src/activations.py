{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jx-yLyGk9IEA"
      },
      "source": [
        "# 과제 - 신경망을 이용한 손글씨 숫자 인식\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UaNKw5VY9IEE"
      },
      "source": [
        "## 1. 환경설정\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "jekllYJU9IEG",
        "outputId": "709ffc28-d8c7-41b9-873f-1897abcbec22",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "GitHub 저장소 URL (예: github.com/USERNAME/mnist-lab.git): github.com/KYUJEONGLEE/Jungle_mnist_lab\n",
            "GitHub Personal Access Token (private 저장소인 경우): ··········\n",
            "Cloning into 'Jungle_mnist_lab'...\n",
            "warning: redirecting to https://github.com/KYUJEONGLEE/Jungle_mnist_lab/\n",
            "remote: Enumerating objects: 108, done.\u001b[K\n",
            "remote: Counting objects: 100% (108/108), done.\u001b[K\n",
            "remote: Compressing objects: 100% (82/82), done.\u001b[K\n",
            "remote: Total 108 (delta 56), reused 71 (delta 24), pack-reused 0 (from 0)\u001b[K\n",
            "Receiving objects: 100% (108/108), 34.21 KiB | 4.28 MiB/s, done.\n",
            "Resolving deltas: 100% (56/56), done.\n"
          ]
        }
      ],
      "source": [
        "# Colab: 이 셀을 가장 먼저 실행하세요 (저장소 클론 후 경로·모듈 로드)\n",
        "# 주의: Colab에서는 GitHub 저장소 URL과 Personal Access Token을 반드시 입력해야 합니다.\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "if \"google.colab\" in sys.modules:\n",
        "    from getpass import getpass\n",
        "\n",
        "    git_url = input(\"GitHub 저장소 URL (예: github.com/USERNAME/mnist-lab.git): \").strip()\n",
        "    token = getpass(\"GitHub Personal Access Token (private 저장소인 경우): \")\n",
        "\n",
        "    # URL 마지막 경로를 저장소 폴더명으로 사용합니다. (예: .../mnist-lab.git -> mnist-lab)\n",
        "    repo_name = Path(git_url.rstrip(\"/\")).name\n",
        "    if repo_name.endswith(\".git\"):\n",
        "        repo_name = repo_name[:-4]\n",
        "\n",
        "    !git clone https://{token}@{git_url}\n",
        "    os.chdir(repo_name)\n",
        "    sys.path.insert(0, str(Path.cwd() / \"src\"))\n",
        "else:\n",
        "    sys.path.insert(0, \"./src\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9p332r2x9IEH"
      },
      "source": [
        "## 2. 데이터 로드"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "BabmiLk_9IEH",
        "outputId": "035699ab-da1f-4824-9fe9-1b83e493547e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Train: (60000, 784) (60000,)\n",
            "Test: (10000, 784) (10000,)\n"
          ]
        }
      ],
      "source": [
        "from data import load_mnist\n",
        "\n",
        "(x_train, y_train), (x_test, y_test) = load_mnist()\n",
        "print('Train:', x_train.shape, y_train.shape)\n",
        "print('Test:', x_test.shape, y_test.shape)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lCXB87na9IEI"
      },
      "source": [
        "## 3. 구현 및 테스트 통과 확인\n",
        "\n",
        "`src/` 아래 역할별 파일의 **TODO**를 순서대로 구현한 뒤 아래 셀을 실행하세요.\n",
        "- 주요 구현 파일: `activations.py`, `layers.py`, `losses.py`, `optimizers.py`, `network.py`, `training.py`\n",
        "- 구현 파일은 역할별 모듈을 직접 import합니다. 예: `from network import NeuralNetwork`\n",
        "- 개발 순서: 과제 안내문 참조\n",
        "- 테스트: `tests/` 아래의 단계별 단위 테스트를 필요한 파일부터 실행합니다. 처음에는 전체 테스트보다 맡은 부분의 테스트 파일을 먼저 실행하세요.\n",
        "    - ReLU만 확인: `TEST_TARGET = \"tests/test_relu.py\"`\n",
        "    - 파일 안의 일부 테스트만 확인: `PYTEST_KEYWORD = \"backward\"`\n",
        "    - 전체 테스트 확인: `TEST_TARGET = \"tests/\"`"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "WiQAydmW9IEI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "69538fae-b653-4d62-8a19-9fd0a2d69bd0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "실행 경로: /content/Jungle_mnist_lab/Jungle_mnist_lab/Jungle_mnist_lab\n",
            "실행 명령: /usr/bin/python3 -m pytest tests/test_relu.py -v\n",
            "============================= test session starts ==============================\n",
            "platform linux -- Python 3.12.13, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3\n",
            "cachedir: .pytest_cache\n",
            "rootdir: /content/Jungle_mnist_lab/Jungle_mnist_lab/Jungle_mnist_lab\n",
            "plugins: anyio-4.13.0, typeguard-4.5.1, langsmith-0.7.34\n",
            "collecting ... collected 3 items\n",
            "\n",
            "tests/test_relu.py::TestReLU::test_relu_forward_positive FAILED          [ 33%]\n",
            "tests/test_relu.py::TestReLU::test_relu_forward_negative_zero FAILED     [ 66%]\n",
            "tests/test_relu.py::TestReLU::test_relu_backward FAILED                  [100%]\n",
            "\n",
            "=================================== FAILURES ===================================\n",
            "_____________________ TestReLU.test_relu_forward_positive ______________________\n",
            "\n",
            "self = <test_relu.TestReLU object at 0x781fb9c2e660>\n",
            "\n",
            "    def test_relu_forward_positive(self):\n",
            "        \"\"\"양수 입력은 ReLU.forward()에서 값이 그대로 유지되어야 한다.\"\"\"\n",
            "        relu = ReLU()\n",
            "        x = np.array([[1.0, 2.0], [3.0, 4.0]])\n",
            ">       out = relu.forward(x)\n",
            "              ^^^^^^^^^^^^^^^\n",
            "\n",
            "tests/test_relu.py:21: \n",
            "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n",
            "\n",
            "self = <activations.ReLU object at 0x781fb9f9b680>\n",
            "x = array([[1., 2.],\n",
            "       [3., 4.]])\n",
            "\n",
            "    def forward(self, x):\n",
            "        \"\"\"\n",
            "        Args:\n",
            "            x: 임의 shape의 입력 배열\n",
            "    \n",
            "        Returns:\n",
            "            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.\n",
            "        \"\"\"\n",
            "        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.\n",
            ">       raise NotImplementedError(\"ReLU.forward를 구현하세요.\")\n",
            "E       NotImplementedError: ReLU.forward를 구현하세요.\n",
            "\n",
            "src/activations.py:30: NotImplementedError\n",
            "___________________ TestReLU.test_relu_forward_negative_zero ___________________\n",
            "\n",
            "self = <test_relu.TestReLU object at 0x781fc2351550>\n",
            "\n",
            "    def test_relu_forward_negative_zero(self):\n",
            "        \"\"\"음수와 0 입력은 ReLU.forward()에서 0으로 바뀌어야 한다.\"\"\"\n",
            "        relu = ReLU()\n",
            "        x = np.array([[-1.0, 2.0], [0.0, -3.0]])\n",
            ">       out = relu.forward(x)\n",
            "              ^^^^^^^^^^^^^^^\n",
            "\n",
            "tests/test_relu.py:28: \n",
            "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n",
            "\n",
            "self = <activations.ReLU object at 0x781fc2337f50>\n",
            "x = array([[-1.,  2.],\n",
            "       [ 0., -3.]])\n",
            "\n",
            "    def forward(self, x):\n",
            "        \"\"\"\n",
            "        Args:\n",
            "            x: 임의 shape의 입력 배열\n",
            "    \n",
            "        Returns:\n",
            "            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.\n",
            "        \"\"\"\n",
            "        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.\n",
            ">       raise NotImplementedError(\"ReLU.forward를 구현하세요.\")\n",
            "E       NotImplementedError: ReLU.forward를 구현하세요.\n",
            "\n",
            "src/activations.py:30: NotImplementedError\n",
            "_________________________ TestReLU.test_relu_backward __________________________\n",
            "\n",
            "self = <test_relu.TestReLU object at 0x781fc230d1f0>\n",
            "\n",
            "    def test_relu_backward(self):\n",
            "        \"\"\"ReLU.backward()는 forward 때 양수였던 위치로만 gradient를 흘려야 한다.\"\"\"\n",
            "        relu = ReLU()\n",
            "        x = np.array([[-1.0, 2.0], [0.0, -3.0]])\n",
            ">       relu.forward(x)\n",
            "\n",
            "tests/test_relu.py:36: \n",
            "_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ \n",
            "\n",
            "self = <activations.ReLU object at 0x781fb9cdd2e0>\n",
            "x = array([[-1.,  2.],\n",
            "       [ 0., -3.]])\n",
            "\n",
            "    def forward(self, x):\n",
            "        \"\"\"\n",
            "        Args:\n",
            "            x: 임의 shape의 입력 배열\n",
            "    \n",
            "        Returns:\n",
            "            x와 같은 shape. x > 0인 위치만 원래 값을 유지합니다.\n",
            "        \"\"\"\n",
            "        # TODO: x > 0 위치를 self.mask에 저장하고, 음수/0 위치는 0으로 바꾸세요.\n",
            ">       raise NotImplementedError(\"ReLU.forward를 구현하세요.\")\n",
            "E       NotImplementedError: ReLU.forward를 구현하세요.\n",
            "\n",
            "src/activations.py:30: NotImplementedError\n",
            "=========================== short test summary info ============================\n",
            "FAILED tests/test_relu.py::TestReLU::test_relu_forward_positive - NotImplemen...\n",
            "FAILED tests/test_relu.py::TestReLU::test_relu_forward_negative_zero - NotImp...\n",
            "FAILED tests/test_relu.py::TestReLU::test_relu_backward - NotImplementedError...\n",
            "============================== 3 failed in 0.20s ===============================\n",
            "\n",
            "\n",
            "선택한 테스트 중 실패가 있습니다.\n"
          ]
        }
      ],
      "source": [
        "import subprocess\n",
        "import sys\n",
        "from pathlib import Path\n",
        "\n",
        "# Colab/로컬 모두 현재 노트북 실행 위치를 저장소 루트로 사용합니다.\n",
        "repo_dir = Path.cwd()\n",
        "\n",
        "# 처음에는 자신이 구현 중인 부분의 테스트 파일만 실행하세요.\n",
        "# 예: tests/test_relu.py, tests/test_affine.py, tests/test_training.py\n",
        "TEST_TARGET = \"tests/test_relu.py\"\n",
        "\n",
        "# 특정 이름이 들어간 테스트만 실행하고 싶을 때 사용합니다.\n",
        "# 예: \"backward\". 전체 파일을 실행하려면 빈 문자열로 둡니다.\n",
        "PYTEST_KEYWORD = \"\"\n",
        "\n",
        "cmd = [sys.executable, \"-m\", \"pytest\", TEST_TARGET, \"-v\"]\n",
        "if PYTEST_KEYWORD:\n",
        "    cmd.extend([\"-k\", PYTEST_KEYWORD])\n",
        "\n",
        "print(\"실행 경로:\", repo_dir)\n",
        "print(\"실행 명령:\", \" \".join(cmd))\n",
        "result = subprocess.run(\n",
        "    cmd,\n",
        "    capture_output=True,\n",
        "    text=True,\n",
        "    cwd=str(repo_dir)\n",
        ")\n",
        "print(result.stdout)\n",
        "if result.stderr:\n",
        "    print(result.stderr)\n",
        "if result.returncode == 0:\n",
        "    print(\"\\n선택한 테스트를 통과했습니다.\")\n",
        "else:\n",
        "    print(\"\\n선택한 테스트 중 실패가 있습니다.\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QlmUeJ0L9IEI"
      },
      "source": [
        "## 4. 모델·옵티마이저 생성 및 학습"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "adnAmDu89IEJ"
      },
      "outputs": [],
      "source": [
        "from network import NeuralNetwork\n",
        "from optimizers import Adam\n",
        "from training import train\n",
        "\n",
        "model = NeuralNetwork(use_batchnorm=True, use_dropout=True)  # BatchNorm, Dropout 필수\n",
        "optimizer = Adam(lr=0.001)\n",
        "\n",
        "loss_history = train(model, optimizer, x_train, y_train, epochs=20, batch_size=128)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GUo3vSS89IEJ"
      },
      "source": [
        "## 5. 평가 및 손실 커브"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-cY60VoE9IEJ"
      },
      "outputs": [],
      "source": [
        "from training import evaluate, plot_loss_history\n",
        "\n",
        "acc, n_params = evaluate(model, x_test, y_test)\n",
        "print(f'Test Accuracy: {acc:.2f}%')\n",
        "print(f'Total Params: {n_params:,}')\n",
        "\n",
        "plot_loss_history(loss_history)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git commit -m \"fuck\"\n",
        "!git push\n",
        "!git config --gl"
      ],
      "metadata": {
        "id": "sGcmEy5OyM9H",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "f17a4d84-2ec6-42a8-91d0-a95d8ba01dc2"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Author identity unknown\n",
            "\n",
            "*** Please tell me who you are.\n",
            "\n",
            "Run\n",
            "\n",
            "  git config --global user.email \"you@example.com\"\n",
            "  git config --global user.name \"Your Name\"\n",
            "\n",
            "to set your account's default identity.\n",
            "Omit --global to set the identity only in this repository.\n",
            "\n",
            "fatal: unable to auto-detect email address (got 'root@7e7d9e3d0105.(none)')\n",
            "fatal: could not read Username for 'https://github.com': No such device or address\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Im7zeinb4bYs"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}