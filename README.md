## 1. WSL
On wsl need install https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
Section with "Installing with Apt"

## 2. Docker
### Here choose only one option docker build or docker pull
```sh 
docker build -t sarna320/twm:latest .
```
```sh 
docker pull  sarna320/twm:latest
```

### You can add a volume by adding for example ```-v ${PWD}/:/home/user/twm/```, to your path
```sh
docker run --gpus all -d --name twm-container sarna320/twm:latest
```

```sh
docker exec -it twm-container bash
```
### Recommendation is to use visual studio code and option to attach to running container

## 3. Check nvidia driver
```sh
nvidia-smi
```

## 4. Check device
```sh
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"
```

## 5. Check in task manger if there is spike ine memory
```sh
python3 -c "import tensorflow as tf;
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
c = tf.matmul(a, b)
print(c)"
```

## 6. Upload photos to test folder and then run:
```sh
python3 main.py
```