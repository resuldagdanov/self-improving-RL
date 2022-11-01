## Installation of Highway Environment

Please follow the [readme instructions](https://github.com/resuldagdanov/self-improving-RL/blob/main/README.md) to install Highway Environment.

<p align="center">
    <img src="../assets/highway-env.gif"><br/>
    <em>An episode of one of the environments available in highway-env</em>
</p>

## Official Highway Environment Documentation

Highway environment is constructed in as a Gym wrapper. For detailed information about the environment, please refer to the [official documentation](https://highway-env.readthedocs.io/en/latest/). Additionally you can find the [source code](https://github.com/eleurent/highway-env).

## Environment Installation

* prepare Ubuntu
```python
sudo apt-get update -y

sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

* install highway-environment
```python
pip install highway-env==1.5
```

* register custom environment wrapper class
```sh
cd highway_environment/highway_environment

python create_env.py
```

* install custom highway environment globally
```sh
cd highway_environment

python setup.py install
```

NOTE: make sure that after each update of Environment class, environment installation has to be performed again.