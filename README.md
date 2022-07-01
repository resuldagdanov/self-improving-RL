# Black-Box-Verification

Test Verification Algorithms on Gym Highway Environment using Ray RLlib

## Anaconda Environment Creation

> used python3.7
```python
conda create -n highway python=3.7

conda activate highway
```

> install required packages
```python
pip install -r requirements.txt
```

## Environment Installation

> prepare Ubuntu
```python
sudo apt-get update -y

sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev
    python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

> install highway-environment
```python
pip install highway-env==1.5
```

> register custom environment wrapper class
```sh
cd highway_environment

python create_env.py
```
