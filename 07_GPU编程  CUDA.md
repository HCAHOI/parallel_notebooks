---
typora-copy-images-to: img
---

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

**注意 : **板块和板块之间,线程和线程之间是高度并行的,不存在板块/线程0的Hello world!就一定打印的比板块/线程1早的情况

### 板块和线程的扁平化

你可能觉得纳闷，既然已经有线程可以并行了，为什么还要引入板块的概念？稍后会说明区分板块的重要原因。

* 如需总的线程数量：blockDim * gridDim

* 如需总的线程编号：blockDim * blockIdx + threadIdx

运行

```cpp
#include<cstdio>
#include<cuda_runtime.h>

__global__ void kernel() {
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tnum = gridDim.x * blockDim.x;
    printf("Flattened Thread %d of %d\n", tid, tnum);
}

int main() {
    kernel<<<2, 3>>>();
    cudaDeviceSynchronize();
	return 0;
}
```

实际上 GPU 的板块相当于 CPU 的线程，GPU 的线程相当于 CPU 的SIMD，可以这样理解，但不完全等同

### 三维的板块和线程编号

CUDA也支持三维的板块和线程区间

只要把三重尖括号内指定的参数改成dim3类型即可,dim3 的构造函数就是接受三个无符号整数（unsigned int）非常简单。

这样在核函数里就可以通过 threadIdx.y 获取 y 方向的线程编号，以此类推。

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
    printf("Block (%d,%d,%d) of (%d,%d,%d), Thread (%d,%d,%d) of (%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z);
}

int main() {
    kernel<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

![image-20220301155124261](.\img\image-20220301155124261.png)

同理,如果需要二维和一维,使用<<<dim3(x,y,1), dim3(m,n,1)>>>和<<<dim3(m,1,1), dim3(n,1,1)>>>

之所以会把板块和线程区分为三维之, 主要是因为 GPU 的业务常常涉及到三维图形学和二维图像，这样很方便，并不一定 GPU 硬件上是三维排列的\

![image-20220301155449439](.\img\image-20220301155449439.png)

### 分离\__device__的声明和定义

默认情况下 GPU 函数必须定义在同一个文件里。如果你试图分离声明和定义，调用另一个文件里的 \__device__ 或 \_ _ global__ 函数，就会出错。

解决方案 : 开启 CMAKE_CUDA_SEPARABLE_COMPILATION, 即可启动分离声明和定义的支持

```cmake
set(CMAKE_CUDA_SEPARABLE_COMLPILATION)
```

但是最好还是把定义和声明放在一个文件里,这样方便编译器自动内联优化(notebook_04)

### 核函数调用核函数

从Kelper架构开始,\_______global______ 里可以调用另一个 \__ global__,也就是说核函数可以调用另一个核函数, 且其三重尖括号里的板块数和线程数可以动态指定，无需先传回到 CPU 再进行调用，这是 CUDA 特有的能力。

```cpp
__global__ void another() {
    printf("another: Thread %d of %d\n", threadIdx.x, blockDim.x);
}

__global__ void kernel() {
    printf("kernel: Thread %d of %d\n", threadIdx.x, blockDim.x);
    int numthreads = threadIdx.x * threadIdx.x + 1;
    another<<<1, numthreads>>>();
    printf("kernel: called another with %d threads\n", numthreads);
}

int main() {
    kernel<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

![image-20220301160900374](.\img\image-20220301160900374.png)

## 内存管理

### 如何从核函数里返回数据

我们试着把kernel里的返回类型声明为int, 试图从GPU返回数据到CPU, 但是这会在编译器出错, 为什么

刚刚说了 kernel 的调用是**异步**的，返回的时候，并不会实际让 GPU 把核函数执行完毕，必须 cudaDeviceSynchronize() 等待他执行完毕（和线程的 join 很像）。所以，不可能从 kernel 里通过返回值获取 GPU 数据，因为 kernel 返回时核函数并没有真正在 GPU 上执行。所以核函数返回类型必须是 void

那如果我们传入一个指针可以吗

```cpp
#include<cstdio>
#include<cuda_runtime.h>

__global__ void kernel(int *pres) {
    *pres = 42;
}

int main() {
    int res = 0;
    kernel<<<1, 1>>>(&res);
    cudaDeviceSynchronize();
    printf("%d", res);
    return 0;
}
```

运行后发现, res打印结果仍然为0

难道是因为我们的 res 创建在栈上，所以 GPU 不能访问，才出错的？

试着用 malloc 在堆上分配一个 int 来给 GPU 访问，结果还是失败了

### 错误检查

CUDA的函数, 比如cudaDeviceSynchronize(), 它们出错时, 并不会直接终止程序, 也不会抛出C++的异常, 而是返回一个错误代码, 告诉你发生了什么错误, 这是处于通用性考虑

这个错误代码的类型是cudaError_t, 其实就是个enum类型,相当于int

通过cudaGetErrorName, 获取该错误代码的具体名字, 这里错误代码是77, 具体名字是cudaErrorIllegalAddress. 意思是说我们访问了非法的地址, 和CPU上的Segmentaton Fault差不多

```cpp
#include<cstdio>
#include<cuda_runtime.h>

__global__ void kernel(int *pres) {
    *pres = 42;
}

int main() {
    int res = 0;
    kernel<<<1, 1>>>(&res);
    cudaError_t err = cudaDeviceSynchronize();
    printf("%d\n", err);
    printf("%s\n", cudaGetErrorName(err));
    
    return 0;
}
```

* 不过, CUDA已经提供了封装的版本

 CUDA toolkit 安装时，会默认附带一系列案例代码，这些案例中提供了一些非常有用的头文件和工具类，比如这个文件：

/opt/cuda/samples/common/inc/helper_cuda.h

把他和 helper_string.h 一起拷到头文件目录里，然后改一下 CMakeLists.txt 让他包含这个头文件目录。

他定义了 checkCudaErrors 这个宏，使用时只需：

```cpp
checkCudaErrors(cudaDeviceSynchronize());
```

即可自动帮你检查错误代码并打印在终端，然后退出。还会报告出错所在的行号，函数名等，很方便。

### 原因分析

原来，GPU 和 CPU 各自使用着独立的内存。CPU 的内存称为主机内存(host)。GPU 使用的内存称为设备内存(device)，他是显卡上板载的，速度更快，又称显存。

而不论栈还是 malloc 分配的都是 CPU 上的内存，所以自然是无法被 GPU 访问到。

因此可以用 cudaMalloc 分配 GPU 上的显存，这样就不出错了，结束时 cudaFree 释放。

注意到 cudaMalloc 的返回值已经用来表示错误代码，所以返回指针只能通过 &pret 二级指针。

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *pret) {
    *pret = 42;
}

int main() {
    int *pret;	//只是在cpu里分配了内存
    checkCudaErrors(cudaMalloc(&pret, sizeof(int)));	//在显存上分配了内存
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    cudafree(pret);
    return 0;
}
```

**反之亦然，CPU也不能访问 GPU 的内存地址**

如果尝试使用

```cpp
printf("%d\n", *pret);
```

打印42,那么将会发现程序抛出了经典段错误segmentation fault

因为这个时候pret是在显存上的, CPU也同样无法访问

### 跨 GPU/CPU 地址空间拷贝数据

可以使用cudaMemory, 它能够在GPU和CPU之间拷贝数据

这里我们希望把GPU上的内存数据拷贝到CPU内存上, 也就是从设备内存(device)到主机内存(host), 因此第四个参数指定为cudaMemcpyDeviceToHost

同理, 还有cudaMemcpyHostToDevice和cudaMemcpyDeviceToDevice

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *pret) {
    *pret = 42;
}

int main() {
    int *pret;
    checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    
    int ret;
    checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d", &ret);
                    
    cudaFree(pret);
    return 0;
}
```

**注意** : cudaMemcpy会自动进行同步操作, 即和cudaDeviceSynchronize()等价,所以上面的cudaDeviceSynchronize()实际上可以删掉了

### 统一内存地址技术

还有一种在比较新的显卡上支持的特性, 统一内存(managed), 只需把cudaMalloc换成cudaMallocManaged即可, 释放时也是通过cudaFreee, 这样分配出来的地址, 不论在CPU上还是在GPU上都是一样的, 都可以访问. 而且拷贝也会自动按需进行(当从CPU访问时), 无需手动调用cudaMemcpy

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *pret) {
	*pret = 42;
}

int main() {
    int *pret;
    checkCudaErrors(cudaMemcpyManaged(&pret, sizeof(int)));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("%d", *pret);
    cudaFree(pret);
    return 0;
}
```

### 总结

主机内存(host) : malloc, free

设备内存(device) : cuadMalloc, cudaFree

统一内存(managed) : cudaMallocManaged, cudaFree

统一内存虽然方便,但并不是完全没有开销, 如果有条件的化还是尽量用分离的主机内存和设备内存吧

## 数组

### 分配数组

和malloc一样,可以用cudaMalloc配合n * sizeof(int), 分配一个大小为n的整型数组, 而 arr 则是指向其起始地址。然后把 arr 指针传入 kernel，即可在里面用 arr[i] 访问他的第 i 个元素。

因为我们用的统一内存(managed)，所以同步以后 CPU 也可以直接读取。

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ kernel(int *arr, int n) {
    for(int i = 0;i < n;i++) {
        arr[i] = 1;
    }
}

int main() {
    int n = 32;
    int *arr;
    checkCudeErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    kernel<<<1, 1>>>(pret, n);
    checkCudaErrors(cudaDeviceSynchronize());
    for(int i = 0;i < n;i++) {
		printf("arr[%d]: %d\n",i, arr[i]);
    }
    cudaFree(arr);
	return 0;
}
```

### 并行赋值

刚刚的for循环是串行的， 我们可以把线程数目调为n， 然后用threadIdx.x作为索引, 这样就实现了每个线程负责给数组中的一个元素赋值

```cpp
__global__ void kernel(int *arr, int n) {
    int i = threadIdx.x;
    arr[i] = 1;
}

int main() {
    int *pret;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    kernel<<<1, n>>>(arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    //...
	return 0;
}
```

### 从线程到板块

核函数内部，用之前说到的 blockDim.x + blockIdx.x + threadIdx.x 来获取线程在整个网格中编号。

外部调用者，则是根据不同的 n 决定板块的数量（gridDim），而每个板块具有的线程数量（blockDim）则是固定的 128。

因此，我们可以用 n / 128 作为 gridDim，这样总的线程数刚好的 n，实现了每个线程负责处理一个元素。

```cpp
__global__ void kernel(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    arr[i] = 1;
}

int main() {
    int n = 65536;
    int *arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));
    
    int nthreads = 128;
    int nblocks = n / nthreads;
    kernel<<<nblocks, nthreads>>>(arr, n);
    
    checkCudaErrors(cudaDeviceSynchronize());
    ...
}
```

但是这样的话n必须是128的整数倍,不然因为C/C++的整数除法是向下取整的,如果n=65535,那么最后127的元素是没有赋值的

解决方案 : 通过数学方法(n + nthreads - 1) / nthreads把这个除法改为向上取整

由于向上取整，这样会多出来一些线程，因此要在 kernel 内判断当前 i 是否超过了 n，如果超过就要提前退出，防止越界。

```cpp
__global__ void kernel(int *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    arr[i] = 1;
}
```

### 小技巧 ：网格跨步循环(grid-stride loop)

无论调用者指定每个板块多少线程（blockDim），总共多少板块（gridDim）。都能自动根据给定的 n 区间循环，不会越界，也不会漏掉几个元素。

这样一个for循环非常符合CPU上常见的parallel_for的习惯, 又能自动匹配不同的blockDim 和 gridDim, 看起来非常方便

```cpp
__global__ void kernel(int *arr, int n) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;i += girdDim.x * blockDim.x) {
        arr[i] = 1;
    }
}
```

##  C++封装

### 抽象的std::allocator接口

你知道吗？std::vector 作为模板类，其实有两个模板参数：std::vector<T, AllocatorT>

那为什么我们平时只用了 std::vector\<T> 呢？因为第二个参数默认是 std::allocator\<T>。

也就是 std::vector\<T> 等价于 std::vector<T, std::allocator\<T>>。

std::allocator\<T> 的功能是负责分配和释放内存，初始化 T 对象等等。

他具有如下几个成员函数：

```cpp
T *allocate(size_t n)            // 分配长度为n，类型为T的数组，返回其起始地址

void deallocate(T *p, size_t n)    // 释放长度为n，起始地址为p，类型为T的数组
```

vector 会调用 std::allocator\<T> 的 allocate/deallocate 成员函数，他又会去调用标准库的 malloc/free 分配和释放内存空间（即他分配是的 CPU 内存）。

我们可以自己定义一个和 std::allocator\<T> 一样具有 allocate/deallocate 成员函数的类，这样就可以“骗过”vector，让他不是在 CPU 内存中分配，而是在 CUDA 的统一内存(managed)上分配。

实际上这种“骗”来魔改类内部行为的操作，正是现代 C++ 的 concept 思想所在。因此替换 allocator 实际上是标准库允许的，因为他提升了标准库的泛用性。

```cpp
template <class T>
struct CudaAllocator {
    using value_type = T;
    
    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }
    
    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
};

__global__ void kernel(int *arr, int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < n;i += blockDim.x * gridDim.x) {
        arr[i] = 1;
    } 
}

int main() {
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    
    kernel<<<32, 128>>>(arr, n);
    
    checkCudaErrors(cudaDeviceSynchronize());
    //...
    return 0;
}
```

### 进一步 : 避免初始化为0

vector 在初始化的时候（或是之后 resize 的时候）会调用所有元素的无参构造函数，对 int 类型来说就是零初始化。然而这个初始化会是在 CPU 上做的，因此我们需要禁用他。

可以通过给 allocator 添加 construct 成员函数，来魔改 vector 对元素的构造。默认情况下他可以有任意多个参数，而如果没有参数则说明是无参构造函数。

因此我们只需要判断是不是有参数，然后是不是传统的 C 语言类型（plain-old-data），如果是，则跳过其无参构造，从而避免在 CPU 上低效的零初始化。

```cpp
template <class T>
struct CudaAllocator {
    using value_type = T;
    
    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }
    
    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
    
    template <class ...Args>
    void construct(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }
};
```

### 进一步 : 核函数可以是模板函数

刚刚说过 CUDA 的优势在于对 C++ 的完全支持。所以 __global__ 修饰的核函数自然也是可以为模板函数的。

调用模板时一样可以用自动参数类型推导，如有手动指定的模板参数（单尖括号）请放在三重尖括号的前面。

```cpp
template<int N, class T>
__global__ void kernel(T *arr) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < N;i += blockDim.x * gridDim.x) {
        arr[i] = 1;
    } 
}

int main() {
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    
    kernel<n><<<32, 128>>>(arr.data());
    
    checkCudaErrors(cudaDeviceSynchronize());
    //...
    return 0;
}
```

### 进一步 : 核函数可以接受函子(functor) , 实现函数式编程

什么是functor

functor不是函数,而是类,但是通过重载()运算符,使其可以像函数一样被调用,所以又被称为仿函数

* 不过要注意三点：

1. 这里的 Func 不可以是 Func const &，那样会变成一个指向 CPU 内存地址的指针，从而出错。所以 CPU 向 GPU 的传参必须按值传。

2. 做参数的这个函数必须是一个有着成员函数 operator() 的类型，即 functor 类。而不能是独立的函数，否则报错。

3. 这个函数必须标记为 __device__，即 GPU 上的函数，否则会变成 CPU 上的函数。

```cpp
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;i += gridDim.x * blockDim.x) {
        func(i);
    }
}

struct MyFunctor {
    __device__ void operator()(int i) const {
        printf("number %d\n", i);
    }
};

int main() {
    int n = 65536;
    
    parallel_for<<<32, 128>>>(n, MyFunctor{});
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    return 0;
}
```

### 进一步 : 函子可以是lambda表达式

可以直接写 lambda 表达式，不过必须在 [] 后，() 前，插入 __device__ 修饰符。

而且需要开启 --extended-lambda 开关。

为了只对 .cu 文件开启这个开关，可以用 CMake 的生成器表达式，限制 flag 只对 CUDA 源码生效，这样可以混合其他 .cpp 文件也不会发生 gcc 报错的情况了

```cmake
target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
```

```cpp
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;i += gridDim.x * blockDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;
    
    parallel_for<<<32, 128>>>(n, [] __device__ (int i) {
        printf("number %d\n", i);
    });
    
    checkCudaErrors(cudaDeviceSynchronize());
    
    return 0;
}
```

### 如何捕获外部变量

#### 尝试一

```cpp
template <class Func>
__global__ void parallel_for(T *arr) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < n;i += blockDim.x * gridDim.x) {
        func(i);
    } 
}

int main() {
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    
   	parallel_for<<<32, 128>>>(n, [&] __device__ (int i) {
       	arr[i] = 1; 
    });
    
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
```

此时,如果单纯的把lambda表达式中的[]改为[&]将会出错,因为此时捕获到的是堆栈(CPU内存)上的变量arr本身, 而不是arr所指向的内存地址(GPU内存)

#### 尝试二

那么,如果我们改为[=]按值捕获呢

错了，不要忘了，vector 的拷贝是深拷贝（绝大多数C++类都是深拷贝，除了智能指针和原始指针）。这样只会把 vector 整个地拷贝到 GPU 上！而不是浅拷贝其起始地址指针。

#### 解决方案

正确的做法是先获取 arr.data() 的值到 arr_data 变量，然后用 [=] 按值捕获 arr_data，函数体里面也通过 arr_data 来访问 arr。

为什么这样？因为 data() 返回一个起始地址的原始指针，而原始指针是浅拷贝的，所以可以拷贝到 GPU 上，让他访问。这样和之前作为核函数参数是一样的，不过是作为 Func 结构体统一传入了。

```cpp
template <class Func>
__global__ void parallel_for(int n, Func func) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < n;i += blockDim.x * gridDim.x) {
        func(i);
    } 
}

int main() {
    int n = 65536;
    std::vector<int,CudaAllocator<int>> arr(n);
    
    int *arr_data = arr.data();
    parallel_for<<<32, 128>>>(n, [=] __device__ (int i) {
       arrdata[i] = i; 
    });
    
    return 0;
}
```

或者在[] 里这样直接写自定义捕获的表达式也是可以的，这样就可以用同一变量名

```cpp
int main() {
    int n = 65536;
    std::vector<int, CudaAllocator<int>> arr(n);
    
   	parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i) {
       	arr_data[i] = i; 
    });
    
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
```

## 数学运算

### 案例 : 并行求sin值

就让我们在 GPU 上并行地计算从 sin(0) 到 sin(65535) 的值，并填入到数组 arr 中。

这里为什么用 sinf 而不是 sin？因为 sin 是 double 类型的正弦函数，而我们需要的 sinf 是 float 类型的正弦函数。可不要偷懒少打一个 f 哦，否则影响性能。

```cpp
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.h"

template<class Func>
__global__ kernel(int n, Func func) {
	for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < n;i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65535;
    std::vector<float,CudaAllocator<float>> arr(n);
    
    kernel<<<32, 128>>>(n, [arr = arr.data()](int i){
        arr[i] = sinf(i);
    });
    
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
```

适当调整板块数量gridDim和线程数目blockDim,还可能更快

注 : 像sinf一样的数学函数还有sqrtf，rsqrtf，cbrtf，rcbrtf，powf，sinf，cosf，sinpif，cospif，sincosf，sincospif，logf，log2f，log10f，expf，exp2f，exp10f，tanf，atanf，asinf，acosf，fmodf，fabsf，fminf，fmaxf。

### __sinf

•两个下划线的 __sinf 是 GPU intrinstics，精度相当于 GLSL 里的那种。适合对精度要求不高，但有性能要求的图形学任务。

•类似的这样的低精度內建函数还有 

```cpp
__expf、__logf、__cosf、__powf 
```

•还有 __fdividef(x, y) 提供更快的浮点除法，和一般的除法有相同的精确度，但是在 2^216 < y < 2^218 时会得到错误的结果。

### 编译器选项 : --use_fast_math

如果开启了 --use_fast_math 选项，那么所有对 sinf 的调用都会自动被替换成 __sinf。

--ftz=true 会把极小数(denormal)退化为0。

--prec-div=false 降低除法的精度换取速度。

--prec-sqrt=false 降低开方的精度换取速度。

--fmad 因为非常重要，所以默认就是开启的，会自动把 a * b + c 优化成乘加(FMA)指令。

开启 --use_fast_math 后会自动开启上述所有

### 案例 : SAXPY

SAXPY(Scalar A times X Plus Y), 即标量 A 乘 X 加 Y

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "CudaAllocator.h"

template<class Func>
__global__ void parallel_for(int i, Func func) {
    for(int i = blockIdx.x * blockDim.x;i < n;i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;
    float a = 1.0f;
    
    std::vector<float, CudaAllocator<float>> x(n);
    std::vector<float, CudaAllocator<float>> y(n);
    
    for(int i = 0;i < n; i++) {
        x[i] = std::rand() + (1.f / RAND_MAX);	//生成[0,1)的浮点数
        y[i] = std::rand() + (1.f / RAND_MAX);
    }
    
    parallel_for<<<32, 128>>>(n, [a, x = x.data(), y = y.data()] __device__ (int i) {
       x[i] = a * x[i] + y[i]; 
    });
    
    checkCudaErrors(cudaDeviceSynchronize());
	return 0;
}
```

## thrust库

### thrust::universal_vector

虽然自己实现CudaAllocator很有趣, 也帮助我们理解了底层原理, 但是既然CUDA官方已经提供了thrust库, 那就用它们的好啦

universal_vector会在统一内存上分配, 因此不论CPU还是GPU都可以访问到

```cpp
thrust::universal_vector<float> x(n);
```

### 分离的device_vector和host_vector

而 device_vector 则是在 GPU 上分配内存，host_vector 在 CPU 上分配内存。

可以通过 = 运算符在 device_vector 和 host_vector 之间拷贝数据，他会自动帮你调用 cudaMemcpy，非常智能。

比如这里的 x_dev = x_host 会把 CPU 内存中的 x_host 拷贝到 GPU 的 x_dev 上。

```cpp
thrust::host_vector<float> x_host(n);
thrust::device_vector<float> x_dev = x_host
```

### 模板函数

#### thrust::generate

thrust 提供了很多类似于标准库里的模板函数，比如 thrust::generate(b, e, f) 对标 std::generate，用于批量调用 f() 生成一系列（通常是随机）数，写入到 [b, e) 区间。

其前两个参数是 device_vector 或 host_vector 的迭代器，可以通过成员函数 begin() 和 end() 得到。第三个参数可以是任意函数，这里用了 lambda 表达式。

```cpp
int n = 65536;
thrust::host_vector<float> x(n);

auto float_rant = [] {
    return rand() / (RAND_MAX + 1.0f);
}

thrust::generate(x.begin(), x.end(), float_rand);
```

#### thrust::for_each

同理还有thrust::for_each(b, e, f)对标std::for_each,他会把 [b, e) 区间的每个元素 x 调用一遍 f(x)。这里的 x 实际上是一个引用。如果 b 和 e 是常值迭代器则是个常引用，可以用 cbegin()，cend() 获取常值迭代器。

```cpp
thrust::device_vector<float> x(n);

thrust::for_each(x.begin(), x.end(), [] __device__ (float &num){
	num += 100.f;
});

thrust::host_vector<float> y = x;

thrust::for_each(y.cbegin(), y.cend(), [] (float const &i) {
	printf("%d\n", i);
});
```

类似对应的函数模板还有 thrust::reduce，thrust::sort，thrust::find_if，thrust::count_if，thrust::reverse，thrust::inclusive_scan 等。

### thrust 模板函数的特点

thrust模板函数可以根据容器类型, 自动决定在GPU还是在CPU运行

for_each 可以用于 device_vector 也可用于 host_vector。当用于 host_vector 时则函数是在 CPU 上执行的，用于 device_vector 时则是在 GPU 上执行的。

这就是为什么我们用于 x 那个 for_each 的 lambda 有修饰，而用于 y 的那个 lambda 需要修饰 \__device__。

### make_zip_iterator

可以用 thrust::make_zip_iterator(a, b) 把多个迭代器合并起来, 相当于Python 里的 zip

然后在函数体里通过 auto const &tup 捕获，并通过 thrust::get\<index>(tup) 获取这个合并迭代器的第 index 个元素……之所以他搞这么复杂，其实是因为 thrust 需要兼容一些“老年程序”爱用的 C++03，不然早该换成 C++11 的 std::tuple 和 C++17 的 structual-binding 语法了

```cpp
thrust::for_each(
	turust::make_zip_iterator(x_dev.begin(), y_dev.cbegin()), 
	thrust::make_zip_iterator(x_dev.end(), y_dev.cend()),
	[a] __device__ (auto const &tup) {
		auto &x = thrust::get<0>(tup);
        auto const &y = thrust::get<1>(tup);
        x = a * x + y;
});
```

## 原子操作

### 案例 : 数组求和

如何并行地对数组进行求和操作 ?

首先让我们试着用串行的思路来解题

因为\__global__函数不能返回值, 只能通过指针. 因此我们先分配一个大小为1的sum数组, 其中的sum[0]用来返回数组的和. 这样同步之后就可以通过sum[0]看到结果了

```cpp
__global__ void parallel_sum (int *sum, int const *arr, int n) {
    for(int i blockDim.x * blockIdx.x + threadIdx.x;i < n;i += blockDi.x * gridDim.x) {
        sum[0] += arr[i];
    }
}

int main() {
    int n = 65536;
    thrust::universal_vector<int> arr(n);
    thrust::universal_vector<int> sum(1);
    
    auto int_rand = [] {
        return std::rand() % 4;
    };
    
    thrust::generate(arr.begin(), arr.end(), int_rand());
    
    TICK(parallel_sum);
    parallel_sum<<<n/128, 128>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_sum);
    
    printf("result: %d\n", sum[0]);
    
    return 0;
}
```

![image-20220302091852423](.\img\image-20220302091852423.png)

却发现结果完全不对, 为什么捏

这是因为GPU上的线程是并行执行的, 然而sum[0] += arr[i] 这个操作,实际上分为四步

1. 读取sum[0]到寄存器A
2. 读取arr[i]到寄存器B
3. 让寄存器A的值加上寄存器B的值
4. 将寄存器A上的值写回sum[0]

```
假如有两个线程分别在 i=0 和 i=1，同时执行：
线程0：读取 sum[0] 到寄存器A（A=0）
线程1：读取 sum[0] 到寄存器A（A=0）
线程0：读取 arr[0] 到寄存器B（B=arr[0]）
线程1：读取 arr[1] 到寄存器B（B=arr[1]）
线程0：让寄存器A加上寄存器B（A=arr[0]）
线程1：让寄存器A加上寄存器B（A=arr[1]）
线程0：写回寄存器A到 sum[0]（sum[0]=arr[0]）
线程1：写回寄存器A到 sum[0]（sum[0]=arr[1]）
这样一来最后 sum[0] 的值是 arr[1]。而不是我们期望的 arr[0] + arr[1]，即算出来的总和变少了！
```

### atomicAdd

这个时候我们就知道, 应该使用atomic了

原子操作的功能就是保证读取/加法/写回三个操作，不会有另一个线程来打扰。

CUDA 也提供了这种函数，即 atomicAdd。效果和 += 一样，不过是原子的。他的第一个参数是个指针，指向要修改的地址。第二个参数是要增加多少。也就是说：

atomicAdd(dst, src) 和 *dst += src 差不多。

```cpp
__global__ void parallel_sum (int *sum, int const *arr, int n) {
    for(int i blockDim.x * blockIdx.x + threadIdx.x;i < n;i += blockDi.x * gridDim.x) {
        atomicAdd(&sum[0], arr[i]);
    }
}
```

atmoicAdd会返回旧值

old = atomicAdd(dst, src) 其实相当于：old = *dst; *dst += src;

利用这一点可以实现往一个全局的数组 res 里追加数据的效果（push_back），其中 sum 起到了记录当前数组大小的作用。

因为返回的旧值就相当于在数组里“分配”到了一个位置一样，不会被别人占据。

```cpp
__global__ parallel_filter(int *sum, int *res, int *arr ,int n) {
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < n;i += blockDim.x * gridDim.x) {
        int loc = atomicAdd(&sum[0], 1);
        res[loc] = arr[i];
    }
}

int main() {
    int n = 1 << 24;
    std::vector<int, CudaAllocator<int>> arr(n);	//src
    std::vector<int, CudaAllocator<int>> sum(1);	//指示器
    std::vector<int, CudaAllocator<int>> res(n);	//dst
    
    for(int i = 0;i < n;i++) {
        arr[i] = std::rand() % 4;
    }
    
    TICK(parallel_filter);
    parallel<<<n/ 4096, 512>>>(sum.data(), res.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
        
    return 0;
}
```

其他的原子操作

```cpp
atomicAdd(dst, src);
    *dst += src;
atomicSub(dst, src);
    *dst -= src;
atomicOr(dst, src);
    *dst |= src;
atomicAnd(dst, src);
    *dst &= src;
atomicXor(dst, src);
	*dst ^= src;
atomicMax(dst, src);
    *dst = std::max(*dst, src);
atomicMin(dst, src);
    *dst = std::min(*dst, src);
```

当然，他们也都会返回旧值（如果需要的话）。

除了之前带有运算的原子操作, 也有这种单纯是写入没有读出的

```cpp
old = atomicExch(dst, src);
```

相当于

```cpp
old = *dst;
*dst = src;
```

*注：Exch是exchange的简写，对标的是std::atomic的exchange函数。*

### atomicCAS

atomicCAS可以原子地判断是否相等, 相等则写入,不相等则返回旧值

```cpp
old = atomicCAS(dst, cmp, src);
```

相当于

```
old = *dst;
if(old == cmp)
	*dst = src;
```

atomicCAS 的作用在于他可以用来实现任意 CUDA 没有提供的原子读-修改-写回指令。比如这里我们通过 atomicCAS 实现了整数 atomicAdd 同样的效果。

```cpp
__device__ __inline__ int my_atmoic_add(int *dst, int src) {
	int old = *dst, expect;
    do {
        expect = old;
        old = atomicCAS(&old, expect, expect + src);
    }while(old != expect);
    return old;
}
```

如果换成expect * src,就变成了原子乘法atomicMul,虽然CUDA没提供,但是我们自己实现了

在老版本的CUDA中的atomicAdd是不支持float的,可以使用CAS配合按位转换(bit-cast)函数\__float_as_int和__int_as_float实现浮点原子加法

```cpp
__device__ __inline__ int float_atomic_add(float *dst, float src) {
	int old = __float_as_int(*dst), expect;
	do {
		expect = old;
		old = atomicCAS((int *)dst, expect, __float_as_int(__int_as_float(expect) + src));
	}while (edpect != old);
	return old;
}
```

### 原子操作的问题 : 影响性能

不过由于原子操作要保证同一时刻只能有一个线程在修改某个地址，如果多个线程同时修改同一个就需要像“排队”那样，一个线程修改完了另一个线程才能进去，非常低效。

解决方案 : 先累加到局部变量 local_sum，最后一次性累加到全局的 sum。

这样每个线程就只有一次原子操作，而不是网格跨步循环的那么多次原子操作了。当然，我们需要调小 gridDim * blockDim 使其远小于 n，这样才能够减少原子操作的次数。比如下面就减少了 4096/512=8 倍的原子操作。

```cpp
__global__ void parallel_sum (int *sum, int const *arr, int n) {
    int local_sum = 0;
    for(int i blockDim.x * blockIdx.x + threadIdx.x;i < n;i += blockDi.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(&sum[0], local_sum);
}

int main() {
    int n = 65536;
    thrust::universal_vector<int> arr(n);
    thrust::universal_vector<int> sum(1);
    
    auto int_rand = [] {
        return std::rand() % 4;
    };
    
    thrust::generate(arr.begin(), arr.end(), int_rand());
    
    TICK(parallel_sum);
    parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(parallel_sum);
    
    printf("result: %d\n", sum[0]);
    
    return 0;
}
```

## 板块与共享内存

到底为什么需要区分出板块的概念

之前说到实际的线程数量就是板块数量(gridDim)乘以每板块线程数量(blockDim)。

那么为什么中间要插一个板块呢？感觉很不直观，不如直接说线程数量不就好了？

### SM（Streaming Multiprocessors）与板块（block）

GPU是由多个流式多处理器(SM)组成的. 每个SM可以处理一个或多个板块

SM又由多个流式单处理器(SP)组成. 每个SP可以处理一个或多个线程

每个SM都有自己的一块共享内存(shared memory), 它的性质类似于CPU中的缓存---和主存相比很小, 但是很快, 用于缓存临时数据. 此外它还有一些特殊的性质, 我们稍后提到.

通常板块的数量总是大于SM的数量, 这时英伟达驱动就会在多个SM之间调度你提交的各个板块. 正如操作系统在多个CPU核心之间轮换调度一样

不过有一点不同, GPU不会像CPU那样做时间片轮换---板块一旦被调度到了一个SM上, 就会一直执行, 直到他执行完退出, 这样的好处是不存在保存和切换上下文(寄存器, 共享内存等)的开销, 毕竟GPU的数据量比较大, 禁不起这样切来切去.

一个SM可以同时运行多个板块, 这时多个板块共用一块共享内存, 每板块分到的内存就少了

而板块内部的线程, 则是被进一步调度到SM上的每个SP

### 无原子的解决方案 : sum变成数组

刚刚的数组求和例子，其实可以不需要原子操作。

首先，声明 sum 为比原数组小 1024 倍的数组。

然后在 GPU 上启动 n / 1024 个线程，每个负责原数组中 1024 个数的求和，然后写入到 sum 的对应元素中去。

因为每个线程都写入了不同的地址，所以不存在任何冲突，也不需要原子操作了。

然后求出的大小为 n / 1024 的数组，已经足够小了，可以直接在 CPU 上完成最终的求和。也就是 GPU 先把数据尺寸缩减 1024 倍到 CPU 可以接受的范围内，然后让 CPU 完成的思路。

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

__global__ void parallel_sum(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum = 0;
        for (int j = i * 1024; j < i * 1024 + 1024; j++) {
            local_sum += arr[j];
        }
        sum[i] = local_sum;
    }
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 1024);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    parallel_sum<<<n / 1024 / 128, 128>>>(sum.data(), arr.data(), n);	//n / 1024 / 128 个板块 每个板块有128个线程 共计 n / 1024 个线程
    checkCudaErrors(cudaDeviceSynchronize());

    int final_sum = 0;
    for (int i = 0; i < n / 1024; i++) {
        final_sum += sum[i];
    }
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}
```

刚刚我们直接用了一个 for 循环迭代所有1024个元素，实际上内部仍然是一个串行的过程，数据是强烈依赖的（local_sum += arr[j] 可以体现出，下一时刻的 local_sum 依赖于上一时刻的 local_sum）

要消除这种依赖，可以通过这样的逐步缩减，这样每个 for 循环内部都是没有数据依赖，从而是可以并行的

```cpp
__global__ void parallel_sum(int *sum, int const *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n / 1024; i += blockDim.x * gridDim.x) {
        int local_sum[1024];
        for (int j = 0; j < 1024; j++) {
            local_sum[j] = arr[i * 1024 + j];
        }
        for (int j = 0; j < 512; j++) {
            local_sum[j] += local_sum[j + 512];
        }
        for (int j = 0; j < 256; j++) {
            local_sum[j] += local_sum[j + 256];
        }
        for (int j = 0; j < 128; j++) {
            local_sum[j] += local_sum[j + 128];
        }
        for (int j = 0; j < 64; j++) {
            local_sum[j] += local_sum[j + 64];
        }
        for (int j = 0; j < 32; j++) {
            local_sum[j] += local_sum[j + 32];
        }
        for (int j = 0; j < 16; j++) {
            local_sum[j] += local_sum[j + 16];
        }
        for (int j = 0; j < 8; j++) {
            local_sum[j] += local_sum[j + 8];
        }
        for (int j = 0; j < 4; j++) {
            local_sum[j] += local_sum[j + 4];
        }
        for (int j = 0; j < 2; j++) {
            local_sum[j] += local_sum[j + 2];
        }
        for (int j = 0; j < 1; j++) {
            local_sum[j] += local_sum[j + 1];
        }
        sum[i] = local_sum[0];
    }
}	//也就是说 比如 去掉数据依赖之后 第一个for循环 i = 0和i = 2和i = 100对应的语句就可以同时进行
```

![image-20220307003516282](.\img\image-20220307003516282.png)

### 板块的共享内存 (shared memory)

刚刚已经实现了无数据依赖可以并行的 for，那么如何把他真正变成并行的呢？这就是板块的作用了，我们可以把刚刚的线程升级为板块，刚刚的 for 升级为线程，然后把刚刚 local_sum 这个线程局部数组升级为板块局部数组。那么如何才能实现**板块局部数组**呢？

同一个板块中的每个线程，都共享着一块存储空间，他就是共享内存。在 CUDA 的语法中，共享内存可以通过定义一个修饰了 __shared__ 的变量来创建。因此我们可以把刚刚的 local_sum 声明为 __shared__ 就可以让他从每个线程有一个，升级为每个板块有一个了。

然后把刚刚的 j 换成板块编号，i 换成线程编号就好啦

```cpp
parallel_sum<<<n / 1024, 1024>>>(sum.data(), arr.data(),n); 
```

```cpp
__global__ void parallel_sum(int *sum, int *arr, int n) {
	__shared__ int localsum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;	//注意顺序 保证内存读取上的连续性
    
    local_sum[j] = arr[i * 1024 + j];
    if(j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    if(j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    if(j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    if(j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    if(j < 32) {
        local_sum[j] += local_sum[j + 32];
    }
    if(j < 16) {
        local_sum[j] += local_sum[j + 16];
    }
    if(j < 8) {
        local_sum[j] += local_sum[j + 8];
    }
    if(j < 4) {
        local_sum[j] += local_sum[j + 4];
    }
    if(j < 2) {
        local_sum[j] += local_sum[j + 2];
    }
    if(j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}
```

但是算出来的结果好像不太对?

这是因为 SM 执行一个板块中的线程时，并不是全部同时执行的。而是一会儿执行这个线程，一会儿执行那个线程。有可能一个线程已经执行到 if (j < 32) 了，而另一个线程还没执行完 if (j < 64)，从而出错。可是为什么 GPU 要这样设计？

因为其中某个线程有可能因为在等待内存数据的抵达，这时大可以切换到另一个线程继续执行计算任务，等这个线程陷入内存等待时，原来那个线程说不定就好了呢？（记得上节课说过内存延迟是阻碍 CPU 性能提升的一大瓶颈，GPU 也是如此。CPU 解决方案是超线程技术，一个物理核提供两个逻辑核，当一个逻辑核陷入内存等待时切换到另一个逻辑核上执行，避免空转。GPU 的解决方法就是单个 SM 执行很多个线程，然后在遇到内存等待时，就自动切换到另一个线程）

因此，我们可以给每个 if 分支后面加上 __syncthreads() 指令。

他的功能是，强制同步当前板块内的所有线程。也就是让所有线程都运行到 __syncthreads() 所在位置以后，才能继续执行下去。

这样就能保证之前其他线程的 local_sum 都已经写入成功了。

```cpp
__global__ void parallel_sum(int *sum, int *arr, int n) {
	__shared__ int localsum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;	//注意顺序 保证内存读取上的连续性
    
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if(j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if(j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if(j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if(j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if(j < 32) {
        local_sum[j] += local_sum[j + 32];
    }
    __syncthreads();
    if(j < 16) {
        local_sum[j] += local_sum[j + 16];
    }
    __syncthreads();
    if(j < 8) {
        local_sum[j] += local_sum[j + 8];
    }
    __syncthreads();
    if(j < 4) {
        local_sum[j] += local_sum[j + 4];
    }
    __syncthreads();
    if(j < 2) {
        local_sum[j] += local_sum[j + 2];
    }
    __syncthreads();
    if(j == 0) {
        sum[i] = local_sum[0] + local_sum[1];
    }
}
```

### 线程组(warp) : 32个线程为一组

其实，SM 对线程的调度是按照 32 个线程为一组来调度的。也就是说，0-31号线程为一组，32-63号线程为一组，以此类推。

因此 SM 的调度无论如何都是对一整个线程组（warp）进行的，不可能出现一个组里只有单独一个线程被调走，要么 32 个线程一起调走。

所以其实 j < 32 之后，就不需要 __syncthreads() 了。因为此时所有访问 local_sum 的线程都在一个组里嘛！反正都是一起调度走，不需要同步。

结果却出错了，难道warp 调度不对？

其实是编译器自作聪明优化了我们对 local_sum 的访问，导致结果不对的。解决：把 local_sum 数组声明为 volatile 禁止编译器优化

```cpp
__shared__ volatile int local_sum[1024];
```

### 线程组分歧(warp divergence)

GPU 线程组（warp）中 32 个线程实际是绑在一起执行的，就像 CPU 的 SIMD 那样。因此如果出现分支（if）语句时，如果 32 个 cond 中有的为真有的为假，则会导致两个分支都被执行！不过在 cond 为假的那几个线程在真分支会避免修改寄存器和访存，产生副作用。而为了避免会产生额外的开销。因此建议 GPU 上的 if 尽可能 32 个线程都处于同一个分支，要么全部真要么全部假，否则实际消耗了两倍时间！

![image-20220307005143285](.\img\image-20220307005143285.png)

那么在之前程序中的条件判断就会影响程序的效率

解决方案: 我们加 if 的初衷是为了节省不必要的运算用的，然而对于 j < 32 以下那几个并没有节省运算（因为分支是按 32 个线程一组的），反而增加了分歧需要避免副作用的开销。因此可以把 j < 32 以下的那几个赋值合并为一个，这样反而快。

```cpp
__global__ void parallel_sum(int *sum, int *arr, int n) {
	__shared__ volatile int local_sum[1024];
    int j = threadIdx.x;
    int i = blockIdx.x;	//注意顺序 保证内存读取上的连续性
    
    local_sum[j] = arr[i * 1024 + j];
    __syncthreads();
    if(j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if(j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if(j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if(j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}
```

### 网格跨步循环版本

共享内存中做求和开销还是有点大，之后那么多次共享内存的访问，前面却只有一次全局内存 arr 的访问，是不是太少了。

因此可以通过网格跨步循环增加每个线程访问 arr 的次数，从而超过共享内存部分的时间。

当然也别忘了在 main 中改变 gridDim 的大小

```cpp
parallel<<<n / 4096, 1024>>>(sum.data(), arr.data(), n);
checkCudaErrors(cudaDeviceSynchronize());
int final_sum = 0;
for(int i = 0;i < n / 4096; i++) {
	final_sum += sum[i];
}
```

```cpp
__global__ void parallel_sum(int *sum, int *arr ,int n) {
	__shared__ volatile int local_sum[1024];
	int j = threadIdx.x;
	int i = blockIdx.x;
	int temp_sum = 0;
	for(int t = i * 1024 + j; t < n; t += 1024 * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if (j < 512) {
        local_sum[j] += local_sum[j + 512];
    }
    __syncthreads();
    if (j < 256) {
        local_sum[j] += local_sum[j + 256];
    }
    __syncthreads();
    if (j < 128) {
        local_sum[j] += local_sum[j + 128];
    }
    __syncthreads();
    if (j < 64) {
        local_sum[j] += local_sum[j + 64];
    }
    __syncthreads();
    if (j < 32) {
        local_sum[j] += local_sum[j + 32];
        local_sum[j] += local_sum[j + 16];
        local_sum[j] += local_sum[j + 8];
        local_sum[j] += local_sum[j + 4];
        local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}
```

### 模板化

我们可以通过模板函数进行包装

```cpp
template <int blockSize, class T>
__global__ void parallel_sum_kernel(T *sum, T const *arr, int n) {
    __shared__ volatile int local_sum[blockSize];
    int j = threadIdx.x;
    int i = blockIdx.x;
    T temp_sum = 0;
    for (int t = i * blockSize + j; t < n; t += blockSize * gridDim.x) {
        temp_sum += arr[t];
    }
    local_sum[j] = temp_sum;
    __syncthreads();
    if constexpr (blockSize >= 1024) {
        if (j < 512)
            local_sum[j] += local_sum[j + 512];
        __syncthreads();
    }
    if constexpr (blockSize >= 512) {
        if (j < 256)
            local_sum[j] += local_sum[j + 256];
        __syncthreads();
    }
    if constexpr (blockSize >= 256) {
        if (j < 128)
            local_sum[j] += local_sum[j + 128];
        __syncthreads();
    }
    if constexpr (blockSize >= 128) {
        if (j < 64)
            local_sum[j] += local_sum[j + 64];
        __syncthreads();
    }
    if (j < 32) {
        if constexpr (blockSize >= 64)
            local_sum[j] += local_sum[j + 32];
        if constexpr (blockSize >= 32)
            local_sum[j] += local_sum[j + 16];
        if constexpr (blockSize >= 16)
            local_sum[j] += local_sum[j + 8];
        if constexpr (blockSize >= 8)
            local_sum[j] += local_sum[j + 4];
        if constexpr (blockSize >= 4)
            local_sum[j] += local_sum[j + 2];
        if (j == 0) {
            sum[i] = local_sum[0] + local_sum[1];
        }
    }
}
```

```cpp
template <int reduceScale = 4096, int blockSize = 256, class T>
int parallel_sum(T const *arr, int n) {
    std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
    parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
    checkCudaErrors(cudaDeviceSynchronize());
    T final_sum = 0;
    for (int i = 0; i < n / reduceScale; i++) {
        final_sum += sum[i];
    }
    return final_sum;
}

int main() {
    int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> sum(n / 4096);

    for (int i = 0; i < n; i++) {
        arr[i] = std::rand() % 4;
    }

    TICK(parallel_sum);
    int final_sum = parallel_sum(arr.data(), n);
    TOCK(parallel_sum);

    printf("result: %d\n", final_sum);

    return 0;
}
```

使用板块局部数组（共享内存）来加速数组求和

这就是 BLS（block-local storage）

### 递归求和

递归地缩并，时间复杂度是 O(logn)。

同样是缩并到一定小的程度开始就切断(cutoff)，开始用 CPU 串行求和。

```cpp
template <int reduceScale = 4096, int blockSize = 256, int cutoffSize = reduceScale * 2, class T>
int parallel_sum(T const *arr, int n) {
    if (n > cutoffSize) {
        std::vector<int, CudaAllocator<int>> sum(n / reduceScale);
        parallel_sum_kernel<blockSize><<<n / reduceScale, blockSize>>>(sum.data(), arr, n);
        return parallel_sum(sum.data(), n / reduceScale);
    } else {
        checkCudaErrors(cudaDeviceSynchronize());
        T final_sum = 0;
        for (int i = 0; i < n; i++) {
            final_sum += arr[i];
        }
        return final_sum;
    }
}
```

### 编译器真聪明口牙!

刚刚说到虽然用了 atomicAdd 按理说是非常低效的，然而实际却没有低效，这是因为编译器自动优化成刚刚用 BLS 的数组求和了！可以看到他优化后的效率和我们的 BLS 相仿，甚至还要快一些！

结论：刚刚我们深入研究了如何 BLS 做数组求和，只是出于学习原理的目的。实际做求和时，直接写 atomicAdd 即可。反正编译器会自动帮我们优化成 BLS，而且他优化得比我们手写的更好

```cpp
__global__ void parallel_sum (int *sum, int const *arr, int n) {
    int local_sum = 0;
    for(int i blockDim.x * blockIdx.x + threadIdx.x;i < n;i += blockDi.x * gridDim.x) {
        local_sum += arr[i];
    }
    atomicAdd(&sum[0], local_sum);
}

int main() {
    int n = 65536;
    thrust::universal_vector<int> arr(n);
    thrust::universal_vector<int> sum(1);
    
    auto int_rand = [] {
        return std::rand() % 4;
    };
    
    thrust::generate(arr.begin(), arr.end(), int_rand());
    
    parallel_sum<<<n / 4096, 512>>>(sum.data(), arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    
    printf("result: %d\n", sum[0]);
    
    return 0;
}
```

## 共享内存进阶
