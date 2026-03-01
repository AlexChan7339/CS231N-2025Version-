4-神经网络与反向传播

> 让神经网络从自身错误中学习

[TOC]



# 复习

## 损失函数

为了构建损失函数，我们需要定义数据集、评分函数($Wx$)。右图定义的就是整个过程。

常用的损失函数：

- softmax
- hingeloss:鼓励正确项($s_j$)的分数高于其他所有项($s_{y_i}$)的分数

铰链损失函数见后图

![image-20260214171132488](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602141711522.png)

![image-20260218213154694](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182131828.png)

我们来看这个式子：$\max(0,\, s_j - s_{y_i} + 1)$，如果我们固定 $s_j$，把它当常数。那么这个式子可以改写成：$\max(0,\, C - s_{y_i})$

其中：$C = s_j + 1$，折点就是$s_{yi}=C=s_j+1$现在它变成一个关于 s_{y_i} 的单变量函数。这就是上图的那条折线。

由于训练的目标是：让正确类别分数变大，让错误类别分数变小；从直觉上看，如果$ s_{y_i}$ 增大，损失会线性下降，直到 margin ≥ 1 为止，所以loss画成：横轴为$s_{y_i}$变量，纵轴为loss变量，此时就能直观看到：正确类别分数越大，loss 越小

## 梯度下降法

山谷上的每一个点对应一组不同的权重参数，希望通过（梯度下降法）损失函数L对W求梯度并对其进行更新（在更新过程中我们定义了步长，其方向一般是沿着梯度负方向进行更新），进而找到使损失景观最小化的参数集W

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602141809288.png" alt="image-20260214180943242" style="zoom:50%;" />

### 数值梯度 VS 解析梯度

为了进行优化，我们讨论了数值梯度和解析梯度两种方法，两者各有优缺点。实际上我们更倾向于使用解析梯度；但是如果数学推导较为困难，我们会采用数值梯度进行验证。

### SGD

对于庞大的数据集来说，计算损失函数和梯度非常复杂。因此我们会采用小批量（32/64/128/256）的数据进行训练

![image-20260214182604198](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602141826231.png)

### 学习率的调整

在训练优化器时，一般从较大的学习率开始，然后使用不同的学习率调整曲线，对学习率进行衰减。但在Adam和AdamW中不需要手动调整学习率。

![image-20260214213055393](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142130431.png)

# 神经网络

> 如何构建神经网络

- 最基本的神经网络：$f=Wx$
  - $C$:输入变量x的维数
  - $D$:输出变量类别数
- 为了构建第二层神经网络，我们需要定义一组新的权重$W_2$，将新的权重应用于前一层神经网络的输出
  - H：神经元隐藏层节点或者神经元的数量
  - $max(·)$：在$W_1$和$W_2$中创造非线性变换
  - b:在实际过程中，我们会加入偏置项（不希望它只过【0,0】点），已建立完整框架。这边为了简洁，我们省略掉bias

![image-20260214213117609](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142131647.png)

## 非线性变换

有很多情况无法通过一条直线分割样本。为了解决这类问题，我们需要某种非线性变换，使样本从原来的空间<x, y>映射到新的空间维度$<r, \theta>$，再运用线性函数进行分割，即用一条线可以分开。

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142133461.png" alt="image-20260214213312430" style="zoom:50%;" />

## 全连接网络

也叫多层感知机（MLPs,单个神经元是单个感知机）,它只依赖权重、输入和层等的网络，层与层之间只有乘法操作，没有其他操作。实际上我们可以堆叠更多层来构建更大规模的网络，后一层与前一层的维度依次进行匹配（对于S层中的神经元来说，每一个神经元都有100个权重）

![image-20260214224218825](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142242685.png)

#### 可视化表示

神经网络中，网络通过权重学习某种模版（$W_1$的每一行相当于一个模版），这些模版是一些图像的代表（取决于训练数据），由是个输出类别生成，并将权重W应用在输入神经元上，h更像得分函数（通常是非线性之后的值），是$W_1$中每个范本出现得分，$W_2$加权所有中间变量得分，得到分类的最终分数

随着层数的增加，我们可以创造更多的模版。下图中，我们有了中间层，可以创造100个模版。通过中间层的100个神经元，我们可以赋予网络生成物体的部分特征（比如眼睛），被所有类别共享。

![image-20260214224248357](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602191147551.png)

#### 激活函数

在神经网络中，我们将$max(·)$称为ReLU激活函数。如果没有激活函数，此时$f=W_2W_1x=W_3x$，其中$W_3=W_2W_1\in R^{C*H},f=W_3x$,此时就可以用矩阵$W_3$代替$W_2W_1$，函数就变成线性函数。所以我们需要某种非线性函数在中间解决非线性问题。

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142308156.png" alt="image-20260214230824117" style="zoom:50%;" />

- ReLU函数（Rectified Linear Unit:修正/整流线性单元）：当x>0时不会出现梯度消失现象；由于它会让一些输出值为0，所以会创造死亡神经元

  - $=max(0, x)$
  - 整流：将x<0的部分抹平（非线性函数）
  - 梯度不会出现饱和状态
  - 由于x>0的梯度可以得到保留，收敛速度上会比sigmoid/tanh快6倍
  - 非常容易计算
  - 输出不是关于0对称，当x<0时，会杀死一部分神经元（梯度为0，不会发生更新）
    - 神经元死亡原因
      - 初始化不良，恰恰让神经元输出和梯度都是0（所以会在初始化时加上偏置项，e.g. 0.01）
      - 学习率太大，会导致在高位空间中振荡，更新到x<0出，无法跳出
  - 对于大多数问题来说，ReLU都是一个好的默认选择

    <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602191255610.png" alt="image-20260219125516536" style="zoom:50%;" />

- Leaky ReLU：避免出现死亡神经元

  - $=max(0.01x, x)$

    - 0.01的$\alpha$值可以自己选定，从x<0的范围里泄露出一个小的ReLU

  - 解决ReLU函数在x<0时，神经元死亡现象，关于0点对称

    <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171516440.png" alt="image-20260217151614376" style="zoom:50%;" />

- ELU：带有指数（用指数拟合饱和【梯度消失】的线）和线性单元，同时以0点为中心

  - Leaky ReLU和ELU都是修正ReLU函数在x<=0时出现死亡神经元的情况，关于0点对称

  - 涉及到指数计算会带来较大的计算量

    <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171517466.png" alt="image-20260217151711434" style="zoom:50%;" />

- Sigmoid / Tanh：他们会将输入值压缩到有限范围内，有时会导致梯度消失（也叫饱和）。因此我们在中间层不使用sigmoid和Tanh，通常将其用于**后期层**，比如要将输出值进行二进制化。

  - Sigmoid函数

    <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602142353299.png" alt="image-20260214235302260" style="zoom:33%;" />

    - 也叫挤压函数，它将输出值挤压到[0,1]，它的梯度是自己 * （1-自己）

    - 可解释性好，可类比神经细胞是否激活（1代表激活，0代表未激活）

    - 不会关于0对称，输出为正数，导致每个权重同向变化（如果关于0对称，则不会出现这个问题）

      下图中$f(·)$为sigmoid函数，$x_i$也为上层sigmoid激活值，那么$f(\sum_iw_ix_i+b)=sigmoid(\sum_iw_ix_i+b)$, $\frac{\partial f(\sum_iw_ix_i+b)}{\partial w_i}=\frac{\partial Sigmoid}{\partial \sum_iw_ix_i+b}\frac{\partial \sum_iw_ix_i+b}{\partial w_i}=\sigma(z)(1-\sigma(z))x_i$为正数（sigmoid函数输出值），而梯度的正负性则取决于$\frac{\partial L}{\partial y}$，当标签 = 1。但预测 0.2时，$\frac{\partial L}{\partial y} <0$；标签 = 0，但预测 0.8，$\frac{\partial L}{\partial y} >0$,因此所有权重更新趋势一起增大或者一起减小

      如果需要$w_1$增大，$w_2$减小，则需要先$w_1, w_2$一起减小($w_2$减少的多一些，$w_1$减少的少一些)，再一起增大（($w_2$增加的少一些，$w_1$增加的多一些)），再一起减小...最终生成锯齿形的优化路径

      <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602161601669.png" alt="image-20260216160118607" style="zoom:50%;" />

      - 指数计算比较消耗计算资源

    - Tanh函数（双曲正切）

      - 可以通过Sigmoid函数缩放平移变化而来
      - 关于0对称，输出有正有负
      - 存在梯度消失问题
      
        <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602191259666.png" alt="image-20260219125912581" style="zoom:50%;" />

## 简单全连接神经网络架构

> 全连接神经网络FC也叫多层感知机MLP

- 定义激活函数，其位置通常放在层中
- 权重W：定义上一层到下一层的权重映射，通过计算点积运算$W·x+b$

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171600111.png" alt="image-20260217160058051" style="zoom:50%;" />

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171601156.png" alt="image-20260217160109115" style="zoom:50%;" />

### python代码实现

- 定义网络
  - N：样本数量
  - D_out：输出维度
  - D_in：输入维度
  - H：隐藏层神经元数量
  - x, y, w1, w2：根据维度随机初始化
- 前向传播
  - 逐层将W应用于输出，最终将预测结果作为预测值
  - 计算损失函数，并在前向传播结束后输出损失值
- 优化
  - <font color=RED>重要：计算解析梯度优化参数w1和w2</font>
  - 对参数w1和w2进行优化

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171614444.png" alt="image-20260217161410389" style="zoom:50%;" />

## 层数和神经元数量不同对分类任务的影响

更多的神经元和层数意味着更强的学习能力，能处理更复杂的函数，更好的区分数据集。神经元和层数过多，会导致过拟合现象，无法泛化到新数据。可以通过正则化防止过拟合，但不要将网络大小放进正则化中进行优化。

> 跟之前最近邻时，K的取值不同，区分数据集的能力（匹配更加复杂的数据集）也不同

![image-20260217164109351](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171641407.png)

$\lambda$控制正则化项在整个损失中的贡献程度，同时正则化项针对权重W的，对其进行约束，产生更加通用性的分类边界，不会产生更加准确的边界细节。因此损失函数是在两部分中实现平衡，一部分是预测正确输出，第二部分是调整权重值，通过平衡得到优秀的分类器。

![image-20260217164125512](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602171641564.png)

## 生物可视化

神经元有一个细胞体，通过树突（dendrite）传递的冲动汇总到细胞体(cell)中,然后利用轴突（axon）将冲动传递到其他神经元中。在函数中，我们会用函数捕捉所有前一层的信号冲动($\sum_iw_ix_i+b$)，在起作用的细胞体中运用激活函数输出激活值($f(\sum_iw_ix_i+b)$)，并将他们传递给下一层的神经元

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181544492.png" alt="image-20260218154411408" style="zoom:50%;" />

生物神经元远比我们建立的神经网络要复杂，我们建立的神经网络框架一般是有规律的，提高计算效率（下图叫3层神经网络或者2层隐藏层神经网络）

![image-20260218154921735](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181549798.png)

![image-20260218154938066](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181549118.png)

## 网络中的损失函数

- 评分函数：通过权重矩阵($W_1, W_2$)将输入转化为评分(s)
- 损失函数：铰链损失函数
- 正则化器
- 损失计算：数据预测损失+正则化器
- 计算L的偏导数，来对参数$W_1, W_2$进行更新

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181558334.png" alt="image-20260218155808283" style="zoom:50%;" />

计算导数往往很繁琐，涉及到大量的矩阵计算；如果需要修改损失函数，需要重新推导进行计算

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181601053.png" alt="image-20260218160124003" style="zoom:50%;" />

## 计算图

有一个更好的办法：计算图+反向传播

- 计算图：将神经网络中的所有操作整合到一起，创建逐步执行的过程。从输入和所有参数开始计算得分函数，将参数放入正则化器中，最终输出损失值。

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181611320.png" alt="image-20260218161121249" style="zoom:50%;" />

大多数构建的神经网络都有图形表示，所有这些复杂函数都可以用同一套框架展示，构建计算图。计算图从图像或输入数据开始，整个网络中有一堆权重，最后通过损失函数输出损失值。

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181618361.png" alt="image-20260218161824310" style="zoom:50%;" />

## 反向传播

> 简单例子
> $$
> f(x,y,z)=(x+y)z
> $$

### 计算图

前向传播：先进行x和y的加法操作得到结果q，再将结果q与z进行相乘

反向传播：利用链式法则(通过q中间变量计算$\frac{\partial f}{\partial x}，\frac{\partial f}{\partial y}$)计算f对每个参数的梯度($\frac{\partial f}{\partial x}，\frac{\partial f}{\partial y},\frac{\partial f}{\partial z}$)

- 上游梯度：从网络末端传来的梯度
- 局部梯度：神经元输出相对于输入的梯度

> 右图中绿色字体表示输入值，红色字体表示该变量的梯度

![image-20260218162333418](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181623514.png)

对于神经元来说，都有输入（x, y）和输出（z）。对于反向传播来说，我们需要局部梯度($\frac{\partial z}{\partial x}，\frac{\partial z}{\partial y}$)和上游梯度（$\frac{\partial L}{\partial z}$）,将上游梯度和局部梯度结合链式法则相乘就可以得到下游梯度，下游梯度将成为前一层的上游梯度

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181722651.png" alt="image-20260218172255555" style="zoom:50%;" />

> 计算loss损失函数对所有变量偏导数作用是，对这些参数朝着梯度反方向进行更新移动，进而找到最优值

---

> 另一个例子
> $$
> f(w,x)=\frac{1}{1+e^{-(w_0x_0+w_1x_1+w_2)}}
> $$
> 下图中绿色值为各个变量的初始值，红色值为梯度值；下方为表达式中一般函数的导数计算表达式

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181734510.png" alt="image-20260218173400393" style="zoom:50%;" />

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181756202.png" alt="image-20260218175610082" style="zoom:50%;" />

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181757882.png" alt="image-20260218175718774" style="zoom:50%;" />

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181758141.png" alt="image-20260218175800033" style="zoom:50%;" />

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181759757.png" alt="image-20260218175943658" style="zoom:50%;" />

<font color=RED>-0.2</font>:$w_0$增加1，输出值减少0.2

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181730880.png" alt="image-20260218173021848" style="zoom:50%;" />

这个例子实质上是将Sigmoid函数作为激活函数，而sigmoid函数的梯度只依赖它自己本身，也就是激活函数的输入值

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181809864.png" alt="image-20260218180902805" style="zoom:50%;" />

### 梯度流模式

- 加法门（梯度分配器）：输入和输出的梯度一样

  <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181813711.png" alt="image-20260218181305635" style="zoom:50%;" />

- 乘积门（交换器）：输入参数的梯度为另一个参数的输入值 （当有两个输入参数时）

  <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181814426.png" alt="image-20260218181454352" style="zoom:50%;" />

- 复制门（梯度加和器）：将网络输出到神经元的梯度进行加和

  <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181818183.png" alt="image-20260218181857111" style="zoom:50%;" />

- 最大值门：将网络输出到神经元的梯度指向输入值的最大值方向

  <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181820380.png" alt="image-20260218182041307" style="zoom:50%;" />

### 反向传播实现

神经网络前向传递计算所有步骤，然后在方向传播是计算梯度

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181827083.png" alt="image-20260218182759025" style="zoom:50%;" />

代码实现上，需要将神经网络的每个函数模块化：创建前向和反向API。下图为乘积门，我们需要在前向传播中存储输入值用于方向传播梯度计算

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181923690.png" alt="image-20260218192356621" style="zoom:50%;" />

### 以向量形式实现

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181940656.png" alt="image-20260218194041588" style="zoom:50%;" />

- $(\frac{\partial y}{\partial x})_n$:y对向量x的第n个元素$x_n$进行求导
- $(\frac{\partial y}{\partial x})_{n,m}$:y的第m个元素$y_m$对向量x的第n个元素$x_n$进行求导
- Solar:标量

---

- loss值始终是标量

- 下图中蓝色值为维度值，$\frac{\partial L}{\partial z}$的维度跟z的维度值一样

- 由于$\frac{\partial z}{\partial x}$的维度是$[D_x,D_z]$,而$\frac{\partial L}{\partial z}$的维度是$D_z$,因此$\frac{\partial L}{\partial x}$的维度是$D_x$,与x的维度保持一致

  <img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602181953847.png" alt="image-20260218195330783" style="zoom:50%;" />

  

### 逐元素的反向传播（4D形式）

输入向量为
$$
\begin{bmatrix}
 1\\
 -2\\
 3\\
-1
\end{bmatrix}
$$
需要为其创建雅可比矩阵（输出的4类对输入的4个值分别的梯度，也是稀疏矩阵），主对角线上元素是否为1取决于输入向量该位置元素输入近ReLU函数中$max(0,x)$是否为x
$$
\begin{bmatrix}
  1&0  &0  &0 \\
  0& 0 &0  &0 \\
  0&0  &1  &0 \\
  0&0  &0  &0
\end{bmatrix}
$$
<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182007316.png" alt="image-20260218200702241" style="zoom:50%;" />

其一般表达式

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182007213.png" alt="image-20260218200724145" style="zoom:50%;" />

### 以矩阵形式实现

- loss值始终是标量
- 下图中蓝色值为维度值，$\frac{\partial L}{\partial z}$的维度跟z的维度值一样
- 由于$\frac{\partial z}{\partial x}$的维度是$[D_x*M_x,D_z*M_z]$,而$\frac{\partial L}{\partial z}$的维度是$[D_z*M_Z]$,因此$\frac{\partial L}{\partial x}$的维度是$[D_x*M_x]$,与x的维度保持一致

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182037490.png" alt="image-20260218203712415" style="zoom:50%;" />

#### 例子

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182038300.png" alt="image-20260218203835230" style="zoom:50%;" />

![image-20260218204328123](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182043212.png)

<img src="https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182045555.png" alt="image-20260218204500506" style="zoom:50%;" />

# 总结

![image-20260218204629950](https://typora-alex2.oss-cn-shanghai.aliyuncs.com/202602182046020.png)