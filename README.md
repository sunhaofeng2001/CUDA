1.命令行下载Anaconda安装包：

```sh
wget https://repo/continuum.io/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-09-03%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:67%;" />

2.安装Anaconda：

```sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-15-53%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:80%;" />

然后有一个阅读的发行说明，一路回车，然后接收，yes：

​	                                            <img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-16-12%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom: 50%;" />

回车确定安装位置：

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-16-20%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:50%;" />

然后初始化，将conda变量加入系统环境，输入yes：

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-17-12%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:50%;" />

重新打开终端或者输入

```shell
source .bashrc
```

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-17-32%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom: 50%;" />

3.创建python的环境变量：

```shell
conda create -n GPU python=3.8
```

其中的-n表示创建一个什么名字的环境变量，GPU是创建的环境变量的名字，python=3.8表示环境的python版本是3.8。

然后输入y：

<img src="https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-18-31%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png" style="zoom:50%;" />

激活环境变量：

```shell
conda activate GPU
```

![2022-07-04 11-19-16 的屏幕截图](https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-19-16%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

查看系统的cuda版本：

```sh
nvidia-smi
```

![2022-07-04 11-20-21 的屏幕截图](https://github.com/sunhaofeng2001/CUDA/blob/master/pic/2022-07-04%2011-20-21%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

看到我的cuda版本是11.7，但是现在的pytorch好像是没有cu11.7的，所以后面使用的是11.6但是不冲突，最好合适的对应的版本。

在pytorch的官网可以看到以下的安装方式：

![屏幕截图 2022-07-04 122710](https://github.com/sunhaofeng2001/CUDA/blob/master/pic/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-07-04%20122710.png)

本来是想要用conda安装，但是conda下载安装太慢了，然后想用别的镜像源安装，结果发现conda公司不允许别人镜像他的包了。感兴趣可以看看怎么用conda镜像安装。所以最后用pip安装，用清华园加速：

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

后面那个cu116代表cuda11.6版本，替换到自己合适的即可。

安装完成后测试：

输入python进入python的命令行：

```
>>>import torch
```

输出cuda是否可用

```
>>>print(torch.cuda.is_available())
```

输出cuda设备数量：

```
>>>print(torch.cuda.device_count())
```

定义变量：

```
>>>x1 = torch.randn(20).cuda(0)
```

```
>>>x2 = torch.randn(20).cuda(0)
```

```
>>>y = x1 + x2
```

```
>>>print(y)
```
继续测试：

```
import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")

ngpu= torch.cuda.device_count()
# Decide which device we want to run on
for num in range(ngpu):
	device = torch.device("cuda:{}".format(num) if (torch.cuda.is_available() and ngpu > 0) else "cpu")
	print("驱动为：",device)
	print("GPU型号： ",torch.cuda.get_device_name(num))
```

```
import 	torch
import  time
print(torch.__version__)
print(torch.cuda.is_available())
# print('hello, world.')


a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
```


