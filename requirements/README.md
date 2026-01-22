# Руководство по установке

Данный гайд поможет вам быстро и правильно настроить рабочее окружение для проекта на **Windows 10/11, Linux и MacOS**.

> **⚠ Важно\!**
> Для работы приложения **необходим Python 3.12**. Версии библиотек подобраны именно для него. Использование Python других версий может привести к ошибкам.

-----

## Шаг 1: Установка Python 3.12

Вам нужен Python 3.12. Выберите свою операционную систему.

### 1\. Windows (10 / 11)

1.  Перейдите на [официальный сайт Python](https://www.python.org/downloads/).
2.  Скачайте установщик для Python 3.12.
3.  Запустите установщик.
4.  **Обязательно** поставьте галочку **"Add Python 3.12 to PATH"** внизу окна установки.
5.  Выберите "Install Now".

### 2\. Linux (Debian / Ubuntu / Fedora)

Рекомендуемый способ — использовать `pyenv`, как в оригинальном руководстве.

```bash
# 1. Установка pyenv
curl https://pyenv.run | bash

# 2. Настройка окружения (добавить в ~/.bashrc или ~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc
# (для Zsh замените .bashrc на .zshrc)

# 3. Установка зависимостей для сборки
# Для Debian/Ubuntu:
sudo apt update && sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev \
tk-dev libffi-dev liblzma-dev
# Для Fedora/RHEL:
sudo dnf install @development-tools zlib-devel bzip2-devel \
readline-devel sqlite-devel openssl-devel tk-devel libffi-devel xz-devel

# 4. Установка Python 3.12
pyenv install 3.12
pyenv global 3.12
```

### 3\. MacOS

Рекомендуемый способ — использовать `pyenv` через Homebrew.

```bash
# 1. Установите Homebrew (если у вас его нет)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Установите pyenv
brew install pyenv

# 3. Настройте окружение (добавить в ~/.zshrc)
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc

# 4. Установка Python 3.12
pyenv install 3.12
pyenv global 3.12
```

-----

## Шаг 2: Создание и активация виртуального окружения

После установки Python 3.12, перейдите в папку проекта в вашем терминале.

### 1\. Windows (Command Prompt или PowerShell)

```bash
# 1. Создайте виртуальное окружение
python -m venv venv

# 2. Активируйте его
.\venv\Scripts\activate
```

### 2\. Linux / MacOS

```bash
# 1. Создайте виртуальное окружение (используя Python 3.12)
python3.12 -m venv venv

# 2. Активируйте его
source venv/bin/activate
```

> **Примечание:** После активации вы должны увидеть `(venv)` в начале строки вашего терминала.

-----

## Шаг 3: Установка PyTorch

Установка PyTorch зависит от вашего оборудования (CPU или GPU).

1.  **Перейдите на официальный сайт PyTorch:** [pytorch.org](https://pytorch.org/get-started/locally/)
2.  **Используйте конфигуратор**, чтобы выбрать параметры для вашей системы (OS, Pip, CUDA/CPU).
3.  **Скопируйте и выполните** сгенерированную команду.

*Примеры команд (могут устареть, всегда проверяйте сайт\!):*

  * **Windows / Linux (CPU):**
    ```bash
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```
  * **Windows (NVIDIA GPU):**
    ```bash
    # Команда будет содержать 'cudaxx' (например, 'cuda118' или 'cuda121')
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
  * **MacOS (Apple Silicon / M1 / M2 / M3):**
    ```bash
    # PyTorch использует 'MPS' для ускорения на чипах Apple
    pip3 install torch torchvision torchaudio
    ```

-----

## Шаг 4: Установка остальных зависимостей

Когда PyTorch установлен, установите все остальные пакеты из файла:

```bash
pip install -r requirements.txt
```

> **Готово\!** Ваше окружение полностью настроено.

-----

## Решение возможных проблем

Если `pip install -r requirements.txt` завершается с ошибкой, попробуйте эти решения.

### **Ошибка: `ModuleNotFoundError: No module named 'tkinter'`**

Эта ошибка означает, что Python не может найти встроенный модуль `tkinter`.

  * **Windows:**
    `tkinter` должен устанавливаться по умолчанию. Если его нет, запустите установщик Python 3.12 еще раз, выберите "Modify" (Изменить) и убедитесь, что установлена галочка **"tcl/tk and IDLE"**.

  * **Linux (Debian/Ubuntu):**

    ```bash
    sudo apt-get install python3.12-tk
    ```

  * **Linux (Fedora/RHEL):**

    ```bash
    sudo dnf install python3-tkinter
    ```

  * **MacOS:**
    Если вы устанавливали Python через `pyenv`, `tkinter` должен быть включен. Если вы использовали Homebrew `python@3.12`, установите `python-tk`:

    ```bash
    brew install python-tk@3.12
    ```

### **Ошибка при установке `scipy` (Отсутствует FORTRAN или BLAS)**

  * **Windows:**
    `pip` должен автоматически скачать "wheel" (.whl). Если он пытается компилировать, убедитесь, что у вас 64-битный Python, и попробуйте обновить `pip`: `pip install --upgrade pip`.

  * **Linux (Debian/Ubuntu):**

    ```bash
    sudo apt-get install gfortran libopenblas-dev
    ```

  * **Linux (Fedora/RHEL):**

    ```bash
    sudo dnf install gcc-gfortran openblas-devel
    ```

  * **MacOS:**
    Установите зависимости через Homebrew:

    ```bash
    brew install openblas gfortran
    ```

После установки этих системных пакетов снова запустите:

```bash
pip install -r requirements.txt
```

### **Проверка установки**

Убедитесь, что всё работает правильно:

```bash
python --version
# Должно быть: Python 3.12.x

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

Ошибка возникает из-за отсутствия Fortran компилятора, который необходим для сборки SciPy из исходного кода. Система пытается найти различные Fortran компиляторы (gfortran, flang, ifort и др.), но не находит ни одного.

## Решения:

### 1. Установите gfortran (рекомендуется)

```bash
# Для Ubuntu/Debian:
sudo apt update
sudo apt install gfortran

# Для CentOS/RHEL/Fedora:
sudo yum install gcc-gfortran
# или
sudo dnf install gcc-gfortran
```

### 2. Установите предварительно собранную версию SciPy

Вместо сборки из исходников можно установить бинарную версию:

```bash
pip install scipy --only-binary=scipy
```

### 3. Используйте conda для установки (если у вас miniconda)

```bash
conda install scipy=1.14.0
```

### 4. Установите системные зависимости для сборки SciPy

Для полной поддержки сборки также рекомендуется установить:

```bash
# Для Ubuntu/Debian:
sudo apt update
sudo apt install build-essential gfortran libopenblas-dev liblapack-dev

# Для CentOS/RHEL/Fedora:
sudo yum groupinstall "Development Tools"
sudo yum install gcc-gfortran openblas-devel lapack-devel
```

### Рекомендуемая последовательность действий:

1. Сначала установите gfortran:
```bash
sudo apt update && sudo apt install gfortran
```

2. Затем повторите установку пакетов:
```bash
pip install albumentations==1.4.10 albucore==0.0.16 opencv-python-headless==4.10.0.84 scipy==1.14.0 matplotlib==3.9.1 ttkbootstrap==1.10.1
```

После установки gfortran сборка SciPy должна пройти успешно.