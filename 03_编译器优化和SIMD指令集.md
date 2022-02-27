## 编译器优化和SIMD指令集

#### **x64** 架构下的寄存器模型

##### x64下的寄存器及功能

![image-20220225080752904](.\img\KoTH3zPYdOI5uap.png)

白色部分为32位架构下的寄存器,灰色为64位新增的寄存器

其中RAX-R15为通用寄存器,esp为堆栈指针寄存器,rax是用来保存返回值的寄存器,RIP记录当前执行的程序地址,MMX,YMM/XMM为存储浮点数的寄存器

可见64位架构下单个寄存器大小从32个字节扩充到了64个字节

![image-20220225081220971](.\img\UBP6p1JmOR9DsFe.png)

一般情况下,函数的前六个参数通过rdi,rsi,rdx,rcx,r8d,r9d传入,存储到堆栈上

##### gcc汇编指令后缀

对于

```cpp
int func(int a ,int b) {	// a ,b分别保存在rdi ,rsi的低32位,即edi ,esi中
	return a * b;
}
```

编译器会将其写为

```assembly
movl	%edi, %edx	# 将edi中的数据(a)移动到edx
imull	%esi, %eax	# 将esi中的数据(b)与eax中的数据(a)相乘,并保存在eax中

ret					# 返回eax中的数据,也就是乘积
```

而当函数改为

```cpp
long long func(long long a ,long long b) {
	return a * b;
}
```

汇编文件将变为

```assembly
movq	%rdi, %rdx	
imulq	%rsi, %rax	

ret					
```

注意二者汇编文件的不同,除去数据存储的空间从较小的edi,edx(32位)变为大一倍的rdi,rax(64位)以外,汇编指令也有所不同,字符后缀从l变为q,这个字符后缀指示了数据的大小,具体如下

| C declaration        | Intel data type  | Assembly-code suffix | Size(bytes) |
| -------------------- | ---------------- | -------------------- | ----------- |
| char                 | Byte             | b                    | 1           |
| short                | Word             | w                    | 2           |
| **int**              | **Double word**  | **l**                | **4**       |
| **long / long long** | **Quad word**    | **q**                | **8**       |
| char *               | Quad word        | q                    | 8           |
| float                | Single precision | s                    | 4           |
| double               | Double Precision | l                    | 8           |

##### 从编译器-O3优化开始

比起基本不改变原有代码机械结构的-Og和-O1优化,-O3优化在提高代码效率的同时,也破坏了代码的可读性

例如

```cpp
int func(int a, int b) {
	return a + b;
}
```

开启优化之后,汇编文件将会被优化为

```assembly
leal	(%rdi,%rsi), %eax

ret
```

其中leal,即lea的功能是加载表达式的地址,并存储在后面标定的位置

(%rdi,%rsi),相当于\*(rdi + rsi),而leal又给原式套上一个&,我们知道取地址符和解析地址符是互逆的,则&\*(rdi + rsi)也就等于rdi + rsi,达成了做加法的目的

同理,return a + 8 * b;将被优化为

```assembly
leal	(%rdi,%rsi,8), %eax
ret
```

##### 指针索引 : size_t

在翻阅网络上的代码时,有时会遇到声明变量时不用int而是size_t的情况,下面将给出理由

例如

```cpp
int func(int *a, int b) {
	return a[b];
}
```

会写为

```assembly
movslq	%esi, %rsi
movl	(%rdi,%rsi,4), %eax

ret
```

其中的movslq是什么意思?原来指针和地址都是64位,而b为32位,在把64位的a加上32位的b之前,就要把b扩充为64位,这里的s可以理解为span

当然,扩充空间的操作必然会影响到程序运行的效率,于是我们可以通过在表示数组大小和索引的时候使用size_t来避免多余的操作

size_t在32位系统等于unit_32,在64位系统上等于uint_64,始终和系统保持同步,也就不需要再使用扩展了,同时64位的size_t也可以处理超过INT_MAX的情况,于是我们**推荐始终用size_t表示数组大小,索引以及循环体的指示变量**

##### 浮点运算与XMM寄存器

当我们进行浮点运算时,将会使用xmm寄存器,xmm寄存器有128位宽,可以容纳4个float或2个double

```cpp
float func(float a, float b) {
	return a + b;
}
```

```assembly
addss	%xmm1, %xmm0

ret
```

其中,addss可以拆分为add, s, s

1.add表示执行加法操作

2.第一个s表示**标量(scalar)**,只对xmm的最低位进行运算;也可以是p表示**矢量(packed)**,一次对xmm中的所有位进行计算

3.第二个s表示**单精度浮点数(single precision)**,即float类型,也可以是d表示**双精度浮点数(double precision)**,即double类型

* addss : 一个float加法

* addsd : 一个double加法

* addps : 四个float加法

* addpd : 两个double加法

  *注: 如果编译器产生的汇编里有大量ss结尾的指令说明矢量化失败,如果大多数都是ps结尾说明矢量化成功*

  

##### SIMD

以上这种**单个指令处理多个数据**的技术称为SIMD(single-instruction multiple-data)

对于处理float的SIMD指令,平均上认为可以加速4倍

#### 编译器优化

##### 代数化简

```cpp
int func(int a, int b) {
    int c = a + b;
    int d = a - b;
    return (c + d) / 2;
}
```

显然b会被消去,所以编译器会直接把a作为返回值,即

```assembly
movl	%edi, %eax
```

还有一种常量化优化

```cpp
int func() {
    int a = 32;
    int b = 10;
    return a + b;
}
```

这里编译器会直接把a与b的和42作为立即数返回

```assembly
movl $42, %eax
```

编译器届的高斯

```cpp
int func(int n) {
    int ret = 0;
    for (int i = 1; i <= 100; i++) {
        ret += i;
    }
    return ret;
}
```

```assembly
movl $5050, %eax
```

然而,对于比较复杂的计算,比如涉及到堆上分配内存的容器,如vector,编译器通常不会进行优化

| 存储在堆上(妨碍优化)                    | 存储在栈上(利于优化)                 |
| --------------------------------------- | ------------------------------------ |
| vector, map, set ,string, function, any | array, bitset, glm::vec, string_view |
| unique_ptr, shared_ptr, weak_ptr        | pair, tuple, optional, variant       |

为什么不全部存在栈上?因为如果存在栈上就无法动态扩充大小, 这就是为什么vector要存在堆上而固定大小的array就可以存在栈上

结论 : 尽量**避免代码复杂化**,避免会造成new/delete的容器

*注: 编译器的优化是有深度限制的,即使全部使用了在栈上的容器,如果在一定时间内无法得到答案的话,编译器就会放弃优化,此时可以在函数定义前加一个constexpr强迫编译器计算完毕,当然,这样会延长编译所用的时间*

##### 内联

对于声明和实现不在同一个文件的函数,编译器无法优化

```cpp
int other (int a);
int func() {
	return other(233);
}
```

只能通过call指令调用外部函数

```assembly
subq	$8, %rsp
movl	$233, %edi
call	_Z5otheri@PLT
addq	$8, %rsp
ret
```

@PLT是Procedure Linkage Table的缩写,即函数链接表.编译器会查找其他.o文件中是否定义了_Z5otheri这个符号,如果定义了,就会把这个@PLT换为相应的地址



如果是声明和定义在同一个文件的内部函数

```cpp
int other (int a) {
	return a;
}
int func() {
	return other(233);
}
```

在普通优化模式下,@PLT将被去掉,因为不需要链接器来实现跳转

在-O3模式下,other()将不会被调用,func将直接返回233

*注: 在正常情况下,由于other()有被其他文件的函数调用的可能,编译器会给other加上_Z5otheri的标签,如果将other()设置为static,派出了暴露给其他函数的可能,那么other()将不会存在于汇编中*

*推荐实验网站: https://godbolt.org*

##### 指针

```cpp
void func(int *a, int *b, int *c) {
	*c = *a;
	*c = *b;
}
```

看上去似乎可以直接优化为*c = *b,但是查看汇编发现并非如此,为什么 ? 如果b和c指向的是同一份地址,前面的优化还正确吗?

这种现象叫做**指针别名(pointer aliasing)**

如果想要完成优化,可以加上gcc附带的restrict关键字,这个关键字在c中是标准的,在cpp中不是标准的,所以需要加上__表示这是当前编译器特定的关键字,它的功能是保证这些指针不会发生重叠

```cpp
void func(int* __restrict a, int* __restrict b, int* __restrict c) {
	*c = *a;
	*c = *b;
}
```

那么我们是否需要给每一个指针都加一个restrict呢,不一定.事实上,restrict只需要加在具有写入访问的指针上,就可以优化成功,其他的指针可以使用const表明禁止写入

```cpp
void func(int const *a, int const *b, int __restrict *c) {
	*c = *a;
	*c = *b;
}
```

如果想要避免编译器对变量的优化,可以使用volatile关键字,可以用于测试性能

restrict和volatile的区别

1. volatile int *a 或 int volatile *a 与 int *__restrict a

2. 语法上区别：volatile 在 * 前面而 __restrict 在 * 后面。

3. 功能上区别：volatile 是禁用优化，__restrict 是帮助优化。

4. 是否属于标准上区别：

* volatile 和 const 一样是 C++ 标准的一部分。

* restrict 是 C99 标准关键字，但不是 C++ 标准的关键字。

* __restrict 其实是编译器的“私货”，好在大多数主流编译器都支持。

##### 矢量化

```cpp
void func(int *a) {
	a[0] = 1;
	a[1] = 2;
}
```

```assembly
movq	.LC0(%rip), %rax
movq	%xmm0, (%rdi)
```

由于a[0],a[1]在这里是一段连续的地址,编译器会使用movq来一次性移动8个字节,从而把两次写入合并为1次

如果为

```cpp
void func(int *a) {
	a[0] = 1;
	a[1] = 2;
}
```

由于不连续,故无法进行相同的优化

同理,四个int32可以合并为一个_m128,这里使用了xmm寄存器

```cpp
void func(int *a) {
	a[0] = 1;
	a[1] = 2;
	a[2] = 3;
	a[3] = 4;
}
```

```assembly
movdqa	.LC0(%rip), %xmm0
movups	%xmm0, (%rdi)
```

movups: move unaligned packed single

u表示(%rdi)的地址不一定对齐到16字节

movaps: move aligned packed single

再同理,八个int32可以合并为一个_m256,按理可以使用ymm寄存器,但是

```cpp
void func(int *a) {
	a[0] = 1;
	a[1] = 2;
	a[2] = 3;
	a[3] = 4;
	a[4] = 5;
	a[5] = 6;
	a[6] = 7;
	a[7] = 8;
}
```

并没有被优化为ymm,而是两个xmm,这是因为编译器不敢保证这台机器支持AVX指令集

在启动编译时,可以使用-march=native来让编译器判断硬件支持的指令,此时如果设备支持,就会使用ymm

```bash
gcc -march=native -O3
```

###### 数组清零

```cpp
void func(int *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}
```

编译器可以自动识别你在干什么,进而直接优化为调用memset,memcpy同理,故不需要为了高效手动改写

###### SIMD加速 : 从0到1024填充

```cpp
void func(int *a) {
    for (int i = 0; i < 1024; i++) {
        a[i] = i;
    }
}
```

通过SIMD可以把以上代码优化为以下等效代码

```cpp
void func(int *a) {
    __m128i curr = {0, 1, 2, 3};
    __m128i delta = {4, 4, 4, 4};
    for (int i = 0; i < 1024; i += 4) {
        a[i : i + 4] = curr;
        curr += delta;
    }
}
```

一次写入4个int,一次计算4个int,显然更加高效惹

数组的大小必须为 4 的整数倍,否则就会写入越界的地址,为了避免这种事情发生,编译器会生成非常庞大的代码,具体为先对前1020个元素用SIMD填入,最后三个元素用传统方法处理,这种方法称为**边界特判法**

如果能保证n是4的倍数,可以在函数开始时加上

```cpp
n = n / 4 * 4;
```

编译器发现n是4的倍数,就不会产生分支了

###### assume_aligned

如果可以保证指针a总能对齐到16字节,在gcc编译器中这样写

```cpp
void func(int *a) {
	n = n / 4 * 4;
    a = (int *)__builtin_assume_aligned(a, 16);
    for (int i = 0; i < 1024; i++) {
        a[i] = i;
    }
}
```

在未来的c20中,将引入std::assume_aligned;但考虑到这种操作优化效率不高,就不需要可以去做了

##### 循环

###### 循环中的指针别名

对于

```cpp
void func(float *a, float *b) {
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] + 1;
    }
}
```

编译器会担心数组a和b有没有指针重合的可能,那样就无法SIMD化,于是将会生成两份汇编代码,一份是 SIMD 的,一份是传统标量的

他在运行时检测 a, b 指针的差是否超过 1024 来判断是否有重叠现象

1.  如果没有重叠，则跳转到 SIMD 版本高效运行。
2. 如果重叠，则跳转到标量版本低效运行，但至少不会错。

解决方案 : 

1. 还是加上__restrict

   ``` cpp
   void func(float *__restrict a, float *__restrict b) {
       for (int i = 0; i < 1024; i++) {
           a[i] = b[i] + 1;
       }
   }
   ```

   这样就只生成SIMD版本了

2. OpenMP强制矢量化

   OpenMP是高性能计算的一个框架,开启方式如下

   ```bash
   gcc -fopenmp -O3
   ```

   启用之后使用如下

   ```cpp
   void func(float *a, float *b) {
   #pragma omp simd
       for (int i = 0; i < 1024; i++) {
           a[i] = b[i] + 1;
       }
   }
   ```

   

###### 循环中的if语句

```cpp
void func(float *__restrict a, float *__restrict b, bool is_mul) {
    for (int i = 0; i < 1024; i++) {
        if (is_mul) {
            a[i] = a[i] * b[i];
        } else {
            a[i] = a[i] + b[i];
        }
    }
}
```

然而,有if分支的循环体是难以SIMD矢量化的,编译器会把if语句移到外面,生成两份代码

也就是优化为以下形式

```cpp
void func(float *__restrict a, float *__restrict b, bool is_mul) {
    if (is_mul) {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] * b[i];
        }
    } else {
        for (int i = 0; i < 1024; i++) {
            a[i] = a[i] + b[i];
        }
    }
}
```

###### 循环中的常量

对于

```cpp
void func(float *__restrict a, float *__restrict b, bool is_mul) {
    for (int i = 0; i < 1024; i++) {
        a[i] = a[i] + b[i] * (dt * dt);
    }
}
```

编译器会把代码尽量从计算次数多的部分(热的)移动到计算次数少的部分(冷的),这里编译器发现dt的值与循环次数无关,dt*dt为常量,就会将其移出循环,化1024次乘为1次乘,提高效率,即相当于

```cpp
void func(float *__restrict a, float *__restrict b, bool is_mul) {
	int dt2 = dt * dt;
    for (int i = 0; i < 1024; i++) {
        a[i] = a[i] + b[i] * dt2;
    }
}
```

需要注意的是,去掉 (dt * dt) 的括号就会优化失败:因为浮点不能精确表示实数,乘法又是左结合的,导致两种计算顺序得到的结果不一定一致,不满足结合律 

###### 另一种SIMD优化失败:调用在另一个文件的函数

```cpp
float func(float *a) {
	float pet = 0;
	for (int i = 0; i < 1024; i++) {
		ret += a[i];
		other();
	}
}
```

编译器看不到那个文件的 other 函数里是什么,哪怕 other 在定义文件里是个空函数,编译器也不敢优化掉

解决方案:写在一个文件里

###### 另一种SIMD优化失败:跳跃/随机访问

```cpp
void func(float *a, int *b) {
    for (int i = 0; i < 1024; i++) {
        a[b[i]] += 1;
    }
}
```

由于b[i]每次不确定,所以不能优化

结论:编译器和CPU都更喜欢顺序的访问

###### 循环展开

```cpp
void func(float *a) {
    for (int i = 0; i < 1024; i++) {
        a[i] = 1;
    }
}
```

每次执行循环体 a[i] = 1后，都要进行一次判断 i < 1024.导致一部分时间花在判断是否结束循环,而不是循环体里

此时我们可以使用unroll展开,例如

```cpp
void func(float *a) {
#pragma GCC unroll 4
    for (int i = 0; i < 1024; i++) {
        a[i] = 1;
    }
}
```

表示将循环体展开为4个,相当于

```cpp
void func(float *a) {
    for (int i = 0; i < 1024; i += 4) {
        a[i + 0] = 1;
        a[i + 1] = 1;
        a[i + 2] = 1;
        a[i + 3] = 1;
    }
}
```

于是减少了循环判断在整个计算过程中的占比

##### 结构体

###### 结构体的矢量化优化

如果结构体中有2个float,对齐到8字节 : 矢量化成功

如果结构体中有3个float,对其到12字节:矢量化失败

原因 : xmm有128位,而三个float为96位,读入x,y,z,x之后,剩下的y和z无法读进同一个xmm中,导致低效

如果将struct设置为

```cpp
struct MyVec {
	float x;
	float y;
	float z;
	char padding[4];
};
```

此时矢量化可以成功了

结论:计算机喜欢2的整数幂,2,4,8,16,32,64,128.结构体大小如果不是2的整数幂,往往导致SIMD优化失败

此外C++11添加了alignas关键字,通过

```cpp
struct alignas(16) MyVec {
	float x;
	float y;
	float z;
};
```

即可将每一个结构体的地址对齐到16字节,就不需要手写padding了

*注: 不一定加上alignas就会变快,这样会导致结构体变大占据更多缓存空间*

###### 结构体的内存布局 : AOS与SOA

* AOS(Array of Struct) : 单个对象的属性紧挨着存 : xyzxyzxyzxyz
* SOA(Struct of Array) : 属性分离存储在多个数组 : xxxxyyyyzzzz

* AOS 必须对齐到 2 的幂才高效，SOA 就不需要。

* AOS 符合直觉，不一定要存储在数组这种线性结构，而 SOA 可能无法保证多个数组大小一致。

* SOA 不符合直觉，但通常是更高效的！

AOS的例子

```cpp
struct MyVec {
	float x;
	float y;
	float z;
};
MyVec a[1024];

void func() {
    for (int i = 0; i < 1024; i++) {
        a[i].x *= a[i].y;
    }
}
```

*符合一般面向对象编程(OOP)习惯,但常常不利于性能*

SOA的例子

```cpp
struct MyVec {
	float x[1024];
	float y[1024];
	float z[1024];
};
MyVec a;

void func() {
    for (int i = 0; i < 1024; i++) {
        a.x[i] *= a.y[i];
    }
}
```

*不符合面向对象编程 (OOP) 的习惯,但常常有利于性能.又称之为面向数据编程 (DOP)*

中间方案 : AOSOA

即4个对象一组打包成SOA,在用一个n/4大小的数组存储AOS

```cpp
struct MyVec {
	float x[4];
	float y[4];
	float z[4];
};
MyVec a[1024/4];

void func() {
    for (int i = 0; i < 1024 / 4; i++) {
        for (int j = 0; j < 4; j++) {
            a[i].x[j] *= a[i].y[j];
        }
    }
}
```

缺点 : 需要两层 for 循环,不利于随机访问;需要数组大小是 4 的整数倍,不过可以用边界特判法解决

##### STL容器

###### std::vector的指针别名问题

将

```cpp
void func(std::vector<int> &a,
          std::vector<int> &b,
          std::vector<int> &c) {
    c[0] = a[0];
    c[0] = b[0];
}
```

改写为

```cpp
void func(std::vector<int> &a,
          std::vector<int> &b,
          std::vector<int> &c) {
    c[0] = a[0];
    c[0] = b[0];
}
```

之后,发现问题仍然存在,因为只有vector被做了restrict,数据是否重合仍然未知

解决方案 : pragma otp simd

###### std::vector的SOA

将AOS的

```cpp
struct MyVec {
    float x;
    float y;
    float z;
};

std::vector<MyVec> a;
```

改写为SOA的

```cpp
struct MyVec {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
};

MyVec a;
```

但是要保证x,y,z三个vector大小相同,一起push一起pop

##### 数学运算

###### 数学优化 :  除法变乘法

编译器会将

```cpp
float func(float a) {
	return a / 2;
}
```

优化为a * 0.5f,根据计算机编码的基础知识,不难想到乘法的速度要快于除法

但是,编译器不会把除以一个数转化为乘这个数的倒数,它害怕这个数等于0

解决方案:

1. 手动取倒数
2. -ffast-math选项让GCC更大胆地尝试浮点运算的优化有时能带来 2 倍左右的提升.作为代价,他对 NaN 和无穷大的处理,可能会和 IEEE 标准规定的不一致.如果能保证程序中永远不会出现 NaN 和无穷大,那么可以打开 -ffast-math

###### 数学函数要加std::前缀

* sqrt 只接受 double

  这导致即使传入的是float参数,也按照double标准计算,相较于float的运算来的更慢

* sqrtf 只接受 float

* std::sqrt** 重载了 double 和 float（推荐）

* abs 只接受 int

*  abs 只接受 double

* fabsf 只接受 float

* std::abs 重载了 int, double, float（推荐）

**总之,请勿用全局的数学函数,他们是 C 语言的遗产.始终用 std::sin, std::pow 等**

###### 嵌套循环的指针别名

```cpp
void func(float *a, float *b, float *c) {
    for (int i = 0; i < 1024; i++) {
        for (int j = 0; j < 1024; j++) {
            c[i] += a[i] * b[j];
        }
    }
}
```

只生成了朴素版本,因为过于复杂了

解决方案:

1. restrict

2. 先读进局部变量,读完后写入

   ```cpp
   void func(float *a, float *b, float *c) {
       for (int i = 0; i < 1024; i++) {
           float tmp = c[i];
           for (int j = 0; j < 1024; j++) {
               tmp += a[i] * b[j];
           }
           c[i] = tmp;
       }
   }
   ```

3. 先加进初始为0的局部变量,再累加到c

   ```
   void func(float *a, float *b, float *c) {
       for (int i = 0; i < 1024; i++) {
           float tmp = 0;
           for (int j = 0; j < 1024; j++) {
               tmp += a[i] * b[j];
           }
           c[i] += tmp;
       }
   }
   ```

   避免浮点数相加导致精度损失(更好)



#### 优化总结

1. 函数尽量写在同一个文件内
2. 避免在for循环内调用外部函数
3. 3.非const指针加上__restrict修饰
4. 试着用SOA取代AOS
5. 对其到16字节或64字节
6. 简单的代码就不要复杂化
7. #pragma omp simd
8. 循环中不变的常量挪到外面
9. 对小循环体使用#pragma unroll
10. -ffast-math 和 -march=native
