# GPU编程 : CUDA

## Hello, world!

### 在CMake中启用支持

最新版的 CMake（3.18 以上），只需在 LANGUAGES 后面加上 CUDA 即可启用。

然后在 add_executable 里直接加你的 .cu 文件，和 .cpp 一样。

```cmake
project(hellocuda LANGUAGES CXX CUDA)

add_executable(main main.cu)
```

### CUDA兼容C++17

CUDA 的语法，基本完全兼容 C++。包括 C++17 新特性，都可以用。甚至可以把任何一个 C++ 项目的文件后缀名全部改成 .cu，都能编译出来。

这是 CUDA 的一大好处，CUDA 和 C++ 的关系就像 C++ 和 C 的关系一样，大部分都兼容，因此能很方便地重用 C++ 现有的任何代码库，引用 C++ 头文件等。

host 代码和 device 代码写在同一个文件内，这是 OpenCL 做不到的。

### hello!

定义函数kernel, 前面加上\__global__修饰符, 即可让它在GPU上运行

```cpp
#include<cstdio>

__global__ void kernel() {
	printf("Hello, world!\n");
}

int main() {
    kernel<<<1, 1>>>();//<<<1, 1>>>是什么意思,之后会说明
    return 0;
}
```

运行之后就可以在GPU上运行printf了, 这里的kernel在GPU上执行,被称为核函数, 用\__global__修饰的就是核函数

然而,如果直接运行的话,是不会打印出Hello, world!的

这是因为 GPU 和 CPU 之间的通信，为了高效，是**异步**的。 CPU 调用 kernel<<<1, 1>>>() 后，并不会立即在 GPU 上执行完毕，再返回。实际上只是把 kernel 这个任务推送到 GPU 的执行队列上，然后立即返回，并不会等待执行完毕。

因此可以调用 cudaDeviceSynchronize()，让 CPU 陷入等待，等 GPU 完成队列的所有任务后再返回。从而能够在 main 退出前等到 kernel 在 GPU 上执行完。

```cpp
int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### 定义在GPU上的设备参数

\__host__将函数定义在了CPU上,因为默认cpp定义的函数就是在CPU上运行的, 所以什么都不加等于加上\__host__

\__global__ 用于定义核函数，他在 GPU 上执行，从 CPU 端通过三重尖括号语法调用，可以有参数，不可以有返回值。

而 \__device__ 则用于定义设备函数，他在 GPU 上执行，但是从 GPU 上调用的，而且不需要三重尖括号，和普通函数用起来一样，可以有参数，有返回值。

即：host 可以调用 global；global 可以调用 device；device 可以调用 device。

```cpp
#include <cstdio>
__device__ say_hello() {
    printf("Hello, world!\n");
}

__global__ kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

此外\_\_global\_\_和  \__host__ 可以同时加在一个函数上 , 这样CPU,GPU都可以调用

### constexpr函数

通过在CMake文件中添加

```cmake
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
```

把 constexpr 函数自动变成修饰 __host__ __device__，从而两边都可以调用。

因为 constexpr 通常都是一些可以内联的函数，数学计算表达式之类的，一个个加上太累了，所以产生了这个需求。

不过必须指定 --expt-relaxed-constexpr 这个选项才能用这个特性，我们可以用 CMake 的生成器表达式来实现只对 .cu 文件开启此选项（不然给到 gcc 就出错了）

当然，constexpr 里没办法调用 printf，也不能用 **syncthreads 之类的 GPU 特有的函数，因此也不能完全替代  \__ host \__  和  \__device__。**

### 通过#ifdef生成不同的代码

CUDA具有多段编译的优点

一段代码他会先送到 CPU 上的编译器（通常是系统自带的编译器比如 gcc 和 msvc）生成 CPU 部分的指令码。然后送到真正的 GPU 编译器生成 GPU 指令码。最后再链接成同一个文件，看起来好像只编译了一次一样，实际上你的代码会被预处理很多次。

他在 GPU 编译模式下会定义 \__CUDA_ARCH__ 这个宏，利用 #ifdef 判断该宏是否定义，就可以判断当前是否处于 GPU 模式，从而实现一个函数针对 GPU 和 CPU 生成两份源码级不同的代码。

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__host__ __global__ void say_hello() {
#ifdef __CUDA_ARCH__
	printf("Hello, world! from GPU\n");
#else
    printf("Hello, world! from CPU\n");
#endif
}

__global__ void kernel() {
    say_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    say_hello();
    return 0;
}
```

### __CUDA_ARCH__

其实 __CUDA_ARCH__ 是一个整数，表示当前编译所针对的 GPU 的架构版本号是多少。这里是 520 表示版本号是 5.2.0，最后一位始终是 0 不用管，我们通常简称他的版本号为 52 就行了。

这个版本号是编译时指定的版本，不是运行时检测到的版本。编译器默认就是最老的 52，能兼容所有 GTX900 以上显卡

可以针对不同的版本号执行不同的操作

```cpp
__host__ __global__ func() {
# if __CUDA_ARCH__ >= 700
	//
#elif __CUDA_ARCH__ >=600
	//
#elif __CUDA_ARCH__ >=500
	//
#elif __CUDA_ARCH__ >=300
	//    
#elif !defined(__CUDA_ARCH__)
	//
}
```

通过CMake,我们可以设置生成对应哪一个架构的指令码,如下

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

对于RTX3050 ，他的版本号是 86，因此最适合他用的指令码版本是 86。

如果不指定，编译器默认的版本号是 52，他是针对 GTX900 系列显卡的。

不过英伟达的架构版本都是向前兼容的，即版本号为 75 的 RTX2080 也可以运行版本号为 52 的指令码，虽然不够优化，但是至少能用。也就是要求：编译期指定的版本 ≤ 运行时显卡的版本。

通过指定多个版本号,也可以生成多份程序,针对不同的架构,将运行不同的程序,但是文件大小和编译时间都会膨胀

**注意 : **在老架构中运行按新架构生成的程序,不仅无法运行,而且不会显式报错

![image-20220301151120741](.\img\image-20220301151120741.png)

## 线程与板块

### 线程

那么我们之前在调用kernel()时,使用的<<<1, 1>>>又是什么意思

如果把<<<1 ,1>>>改为<<<1, 3>>>,将会发现Hello, world!被打印了三遍

原来<<<1, 1>>>中的第二个1指定了启动kernel时所用的GPU的线程数量

我们可以通过threadIdx.x获取当前线程的编号, 用blockDim.x获取当前线程的数目

```cpp
__global__ void kernel() {
	printf("Thread %d of %d\n", threadIdx.x, blockDim.x);
}
```

```
Thread 0 of 3
Thread 1 of 3
Thread 2 of 3
```

这两个变量是CUDA中的特殊变量, 只有在核函数中才能访问,至于这里的.x,后面再做说明

### 板块

CUDA中还有比线程更大的概念,板块(block)，一个板块可以有多个线程组成。这就是为什么刚刚获取线程数量的变量用的是 blockDim，实际上 blockDim 的含义是每个板块有多少个线程。

要指定板块的数量，只需调节三重尖括号里第一个参数即可。我们这里调成 2。总之：

**<<<板块数量，每个板块中的线程数量>>>**

而板块的编号通过blockIdx.x获取, 板块的总数通过gridDim获取
