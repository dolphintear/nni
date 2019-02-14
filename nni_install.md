# NNI 安装记录与使用心得

- NNI 项目地址：[https://github.com/Microsoft/nni](https://github.com/Microsoft/nni)
- 运行环境：Windows Subsystem for Linux，python3.6.7

## 安装

1. 通过```pip install nni```命令安装  
![安装](imgs/2019-02-14-19-00-39.png)
2. 下载NNI源码```git clone -b v0.5.1 https://github.com/Microsoft/nni.git```  
![下载NNI源码](imgs/2019-02-14-19-25-26.png)
3. 运行mnist示例```nnictl create --config nni/examples/trials/mnist/config.yml```  
![运行mnist示例](imgs/2019-02-14-19-37-17.png)
4. 实验结果 & Web UI  
![运行结果](imgs/2019-02-14-21-47-32.png)
不同Trial的结果按性能排序，点击Trial No.旁的加号可以展开该Trial所使用的一组参数。  
![运行结果](imgs/2019-02-14-21-48-18.png)
点击导航栏的Trials Detail按钮，可以得到实验详情，包括每组超参数的性能、各个超参数对性能的影响、每组参数的运行耗时等。  
![每组超参数的性能](imgs/2019-02-14-22-01-20.png)  
![各个超参数对性能的影响](imgs/2019-02-14-22-01-47.png)  
![每组参数的运行耗时](imgs/2019-02-14-22-02-24.png)  
另外，还可以查看每组参数运行的中间结果，手动终止Trial的运行：  
![控制面板](imgs/2019-02-14-22-05-47.png)
![中间过程](imgs/2019-02-14-22-06-33.png)

## 代码分析
使用```colordiff nni/examples/trials/mnist/mnist_before.py nni/examples/trials/mnist/mnist.py```命令对比未使用NNI的mnist_before.py和使用nni的mnist.py文件可知，NNI框架对原代码的改动很少，只新增了5行必要代码。  
![代码对比](imgs/2019-02-14-22-14-53.png)

以下是nni训练配置文件，可以配置最大训练时长、最大训练次数、调参算法等等，具有很大的灵活性。

```yaml
#config.yml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
```

以下是search_space.json文件，定义了参数搜索空间。根据以下配置文件可以产生 2 * 4 * 3 * 5 * 4 = 480 种组合！NNI通过内置的参数调优算法，缩小了参数搜索空间，在本次实验中只尝试了9种参数组合就得到了0.97的高分，节约了时间，也省去了人工调参的麻烦。

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
    "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
}
```

## 与其它自动机器学习工具的比较

### Scikit-Optimize

[Scikit-Optimize，简称skopt](https://github.com/scikit-optimize/scikit-optimize)是一个超参数优化库，包括随机搜索、贝叶斯搜索、决策森林和梯度提升树等，用于辅助寻找机器学习算法中的最优超参数。以下是skopt的示例代码，给定参数范围，skopt找出最优参数。

```python
import numpy as np
from skopt import gp_minimize

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) * np.random.randn() * 0.1)

res = gp_minimize(f, [(-2.0, 2.0)])
```

相比NNI，skopt只提供单纯的超参数搜索功能，不提供Web UI管理功能，分布式环境和GPU支持也需要自行配置，不适合工程应用场景。

### hyperopt-sklearn

[hyperopt-sklearn]是一个基于Hyperopt的scikit-learn机器学习算法模型选择框架，提供超参选择功能。相比NNI框架，除了无Web UI管理功能、不原生支持分布式环境和GPU，安装也较为复杂，以下是安装命令：

```shell
git clone git@github.com:hyperopt/hyperopt-sklearn.git
(cd hyperopt-sklearn && pip install -e .)
```

通过以下示例代码可以看出，hyperopt-sklearn对原有代码的改动较大，超参搜索空间也采用硬编码的方式给出，不利于代码维护和阅读。

```python
from hpsklearn import HyperoptEstimator, sgd
from hyperopt import hp
import numpy as np

sgd_penalty = 'l2'
sgd_loss = hp.pchoice(’loss’, [(0.50, ’hinge’), (0.25, ’log’), (0.25, ’huber’)])
sgd_alpha = hp.loguniform(’alpha’, low=np.log(1e-5), high=np.log(1))

estim = HyperoptEstimator(classifier=sgd(’my_sgd’, penalty=sgd_penalty, loss=sgd_loss, alpha=sgd_alpha))
estim.fit(X_train, y_train)
```

### Simple(x) Global Optimization

[Simple(x) Global Optimization](https://github.com/chrisstroemel/Simple)是一个优化库，可作为贝叶斯优化的替代方法。它和贝叶斯搜索一样，试图以尽可能少的样本进行优化，但也将计算复杂度从n降低到log(n)，这对大型搜索空间非常有用。这个库使用单形（n维三角形），而不是超立方体（n维立方体），来模拟搜索空间，这样做可以避开贝叶斯优化中具有高计算成本的高斯过程。  
![对比](https://github.com/chrisstroemel/Simple/raw/master/comparison.gif?raw=true)
和上述两种工具的缺点一样，都不支持Web UI管理、分布式环境、GPU支持，对原代码的改动也很大：

```python
from Simple import SimpleTuner

objective_function = lambda vector: -((vector[0] - 0.2) ** 2.0 + (vector[1] - 0.3) ** 2.0) ** 0.5
optimization_domain_vertices = [[0.0, 0.0], [0, 1.0], [1.0, 0.0]]
number_of_iterations = 30
exploration = 0.05 # optional, default 0.15

tuner = SimpleTuner(optimization_domain_vertices, objective_function, exploration_preference=exploration)
tuner.optimize(number_of_iterations)
best_objective_value, best_coords = tuner.get_best()

print('Best objective value ', best_objective_value)
print('Found at sample coords ', best_coords)
tuner.plot() # only works in 2D
```

### Chocolate

[Chocolate](https://github.com/AIworx-Labs/chocolate)是一个完全异步的优化框架，仅依靠数据库在工作者之间共享信息。Chocolate不使用主过程来分配任务。每个任务都是完全独立的，只能从数据库中获取信息。因此Chocolate在难以维持主过程的受控计算环境中是理想的。示例代码如下：

```python
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

space = {"x" : choco.uniform(-6, 6),
         "y" : choco.uniform(-6, 6)}
conn = choco.SQLiteConnection("sqlite:///my_db.db")
sampler = choco.QuasiRandom(conn, space, random_state=42, skip=0)
token, params = sampler.next()
loss = himmelblau(**params)
sampler.update(token, loss)

>>> print(token)#token用于追踪状态
{"_chocolate_id" : 0}
```

相比NNI，Chocolate也支持分布式环境，但应用场景有限，管理也不方便。

## 总结

NNI工具只需一条命令即可安装，原有机器学习代码只需简单的改动即可利用NNI框架自动选择最佳超参数，配置简单灵活，提供可视化管理界面，易于上手。希望NNI工具能早日支持Windows系统，日常应用软件只有Windows版，使用NNI却要切换到Linux系统，带来不必要的麻烦。