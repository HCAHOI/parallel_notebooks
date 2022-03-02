# 编译器, CMake和Git

如果第一次来到这里，可以选择不看，我相信你会回来的（笑）

 ## 编译器

### 简介

什么是编译器

**编译器**， 是一个根据**源代码**生成**机器码**的程序

以linux平台上的gdb为例(windows上的mingw即为gdb的移植版本) 

我们在完成以下的代码之后

```cpp
//main.cpp
#include<iostream>

int main() {
    printf("Hello, world!\n");
	return 0;
}
```

运行

```bash
g++ main.cpp -o a.out
```

该命令会调用编译器程序g++， 让他读取mian.cpp中的源码， 并根据C++标准生成相应的机器指令码， 输出到a.out这个文件中（称为可执行文件）,其中g++为C++编译器, gcc为C语言编译器

```bash
./a.out
```

之后执行该命令，操作系统会读取刚刚生成的可执行文件，从而执行其中编译成机器码，调用系统提供的printf函数，并在终端显示出Hello, world。

几个基本指令

* -o 表示编译并链接

* -c 表示只进行编译

* -Ox表示优化等级 -Og 不优化 -O1, -O2, -O3分别表示三种不同的优化等级, 数字越大等级越高, 反汇编出的代码和源码相差越大

于是以下命令

```bash
g++ main.cpp -O3 -c main.o
```

的意思就很显然了

### 多文件编译与链接

单文件编译虽然方便，但也有如下缺点：

1. 所有的代码都堆在一起，不利于模块化和理解。

2. 工程变大时，编译时间变得很长，改动一个地方就得全部重新编译。

因此，我们提出多文件编译的概念，文件之间通过**符号声明**相互引用。

```bash
g++ -c hello.cpp -o hello.o
g++ -c main.cpp -o main.o
```

其中使用 -c 选项指定生成临时的**对象文件** main.o，之后再根据一系列对象文件进行链接，得到最终的a.out：

```bash
g++ hello.o main.o -o a.out
```

## 构建系统 : Makefile

### 为什么需要构建系统

文件越来越多时，一个个调用g++编译链接会变得很麻烦。

于是，发明了 make 这个程序，你只需写出不同文件之间的**依赖关系**，和生成各文件的规则。

在当前目录定义Makefile文件

```makefile
a.out: hello.o main.o
	g++ hello.o main.o -o a.out
hello.o: hello.cpp
	g++ -c hello.cpp -o hello.o
main.o: main.cpp
	g++ -c main.cpp -o main.o
```

*typora的语法高亮还挺高级*

写好之后敲下

```bash
make a.out
```

就可以构建出a.out这个可执行文件了

和直接用一个脚本写出完整的构建过程相比，make 指明依赖关系的好处：

1. 当更新了hello.cpp时只会重新编译hello.o，而不需要把main.o也重新编译一遍。

2. 能够自动**并行**地发起对hello.cpp和main.cpp的编译，加快编译速度（make -j）。

3. 用通配符批量生成构建规则，避免针对每个.cpp和.o重复写 g++ 命令（%.o: %.cpp）。

但坏处也很明显：

1. make 在 Unix 类系统上是通用的，但在 Windows 则不然。

2. 需要准确地指明每个项目之间的依赖关系，有头文件时特别头疼。

3. make 的语法非常简单，不像 shell 或 python 可以做很多判断等。

4. 不同的编译器有不同的 flag 规则，为 g++ 准备的参数可能对 MSVC 不适用。

## CMAKE : 构建系统的构建系统

> 众王之王, 众哈里发的哈里发()

### 简介

为了解决 make 的以上问题，跨平台的 CMake 应运而生！

~~make 在 Unix 类系统上是通用的，但在 Windows 则不然。~~

只需要写一份 **CMakeLists.txt**，他就能够在调用时**生成**当前系统所支持的构建系统。

~~需要准确地指明每个项目之间的依赖关系，有头文件时特别头疼。~~

CMake 可以自动检测源文件和头文件之间的依赖关系，导出到 Makefile 里。

~~make 的语法非常简单，不像 shell 或 python 可以做很多判断等。~~

CMake 具有相对高级的语法，**内置的函数**能够处理 configure，install 等常见需求。

~~不同的编译器有不同的 flag 规则，为 g++ 准备的参数可能对 MSVC 不适用。~~

CMake 可以自动检测当前的编译器，需要添加哪些 flag。比如 OpenMP，只需要在 CMakeLists.txt 中指明 target_link_libraries(a.out OpenMP::OpenMP_CXX) 即可。

框架

```cmake
cmake_minimum_required(VERSION 3.18)

project(hello)

set(CMAKE_CXX_STANDARD 17)	#设置C++标准

add_executable(main main.cpp) 	#前面为输出的可执行文件 后面为输入的源文件
```

### CMake的命令行调用

读取当前目录的 CMakeLists.txt，并在 build 文件夹下生成 build/Makefile

```bash
cmake -B build
```

让 make 读取 build/Makefile

```bash
make -C build
```

a.out将会生成在当前目录或./build目录

之后每次修改CMakeLists.txt之后运行

```bash
cmake .
```

修改源码运行

```bash
make
```

即可

### 库

为什么需要库

有时候我们会有多个可执行文件，他们之间用到的某些功能是相同的，我们想把这些共用的功能做成一个**库**，方便大家一起共享。

库中的函数可以被可执行文件调用，也可以被其他库文件调用。

库文件又分为**静态库文件**和**动态库文件**。

* 静态库( .a )相当于直接把代码插入到生成的可执行文件中，会导致体积变大，但是只需要一个文件即可运行。

* 动态库( .so )则只在生成的可执行文件中生成“插桩”函数，当可执行文件被加载时会读取指定目录中的.dll文件，加载到内存中空闲的位置，并且替换相应的“插桩”指向的地址为加载后的地址，这个过程称为**重定向**。这样以后函数被调用就会跳转到动态加载的地址去。

Windows：可执行文件同目录，其次是环境变量%PATH%

Linux：ELF格式可执行文件的RPATH，其次是/usr/lib等

### CMake中的静态库和动态库

CMake 除了 add_executable 可以生成**可执行文件**外，还可以通过 add_library 生成**库文件**。

add_library 的语法与 add_executable 大致相同，除了他需要指定是**动态库**还是**静态库**：

```cmake
add_library(libtest STATIC source1.cpp source2.cpp)	#生成静态库 libtest.a
add_library(libtest SHARED source1.cpp source2.cpp)	#生成动态库 libtest.so
```

动态库有很多坑，特别是 Windows 环境下，初学者自己创建库时，建议使用静态库。

但是他人提供的库，大多是作为动态库的，我们之后会讨论如何使用他人的库。

创建库以后，要在某个**可执行文件**中使用该库，只需要：

```cmake
target_link_libraries(myexec PUBLIC libtest)	#为myexec链接刚刚制作的库libtest.a
```

其中 PUBLIC 的含义稍后会说明（CMake 中有很多这样的大写修饰符）

### CMake中的子模块

复杂的工程中，我们需要划分子模块，通常一个库一个目录，比如：

![image-20220302123032242](.\img\image-20220302123032242.png)

这里我们把 hellolib 库的东西移到 hellolib 文件夹下了，里面的 CMakeLists.txt 定义了 hellolib 的生成规则。

要在根目录使用他，可以在外面用 CMake 的 add_subdirectory 添加子目录，子目录也包含一个 CMakeLists.txt 用add_library定义一个库 ，其中定义的库在 add_subdirectory 之后就可以在外面使用。

子目录的 CMakeLists.txt 里路径名（比如 hello.cpp）都是相对路径，这也是很方便的一点

**子模块的头文件如何处理**

因为 hello.h 被移到了 hellolib 子文件夹里，因此 main.cpp 里也要改成:

```cpp
#include "hellolib/hello.h"
```

如果要避免修改代码，我们可以通过 target_include_directories 指定a.out 的头文件搜索目录：(其中第一个 hellolib 是库名，第二个是目录)

```cmake
add_executable(a.out main.cpp)
target_link_libraries(a.out PUBLIC hellolib)
target_include_directories(a.out PUBLIC hellolib)
```

这样甚至可以用 <hello.h> 来引用这个头文件了，因为通过 target_include_directories 指定的路径会被视为与系统路径等价

那么,如果另一个b.out也要用hellolib这个库, 难道也得再指定一遍路径吗?

不需要,  其实我们只需要定义 hellolib 的头文件搜索路径，引用他的可执行文件 CMake 会**自动添加这个路径**：

![image-20220302124322196](.\img\image-20220302124322196.png)

这里用了 . 表示当前路径，因为子目录里的路径是相对路径，类似还有 .. 表示上一层目录。

此外，如果不希望让引用 hellolib 的可执行文件自动添加这个路径，把 **PUBLIC** 改成 **PRIVATE** 即可。这就是他们的用途：决定一个属性要不要在被 link 的时候传播。

### 一些其他选项

```cmake
target_include_directories(myapp PUBLIC /usr/include/eigen3)  	# 添加头文件搜索目录
target_link_libraries(myapp PUBLIC hellolib)               		# 添加要链接的库
target_add_definitions(myapp PUBLIC MY_MACRO=1)             	# 添加一个宏定义
target_add_definitions(myapp PUBLIC -DMY_MACRO=1)         		# 与 MY_MACRO=1 等价
target_compile_options(myapp PUBLIC -fopenmp)               	# 添加编译器命令行选项
target_sources(myapp PUBLIC hello.cpp other.cpp)             	# 添加要编译的源文件
```

其中PUBLIC和PRIVATE的使用和之前是相同的

### 第三方库

有时候我们不满足于 C++ 标准库的功能，难免会用到一些第三方库。

* 最友好的一类库莫过于纯头文件库了，这里是一些好用的 header-only 库

1. nothings/stb - 大名鼎鼎的 stb_image 系列，涵盖图像，声音，字体等，只需单头文件！

2. Neargye/magic_enum - 枚举类型的反射，如枚举转字符串等（实现方式很巧妙）

3. g-truc/glm - 模仿 GLSL 语法的数学矢量/矩阵库（附带一些常用函数，随机数生成等）

4. Tencent/rapidjson - 单纯的 JSON 库，甚至没依赖 STL（可定制性高，工程美学经典）

5. ericniebler/range-v3 - C++20 ranges 库就是受到他启发（完全是头文件组成）

6. fmtlib/fmt - 格式化库，提供 std::format 的替代品（需要 -DFMT_HEADER_ONLY）

7. gabime/spdlog - 能适配控制台，安卓等多后端的日志库（和 fmt 冲突！）

只需要把他们的 include 目录或头文件下载下来，然后 include_directories 即可。

缺点：函数直接实现在头文件里，没有提前编译，从而需要重复编译同样内容，编译时间长。

* 第二友好的方式则是作为 CMake 子模块引入，也就是通过 add_subdirectory。

方法就是把那个项目（以fmt为例）的源码放到你工程的根目录, 那么你的根目录下会有一个名为fmt的文件夹

```cmake
add_subdirectory(fmt)
```

这些库能够很好地支持作为子模块引入：

1. fmtlib/fmt - 格式化库，提供 std::format 的替代品

2. gabime/spdlog - 能适配控制台，安卓等多后端的日志库

3. ericniebler/range-v3 - C++20 ranges 库就是受到他启发

4. g-truc/glm - 模仿 GLSL 语法的数学矢量/矩阵库

5. abseil/abseil-cpp - 旨在补充标准库没有的常用功能

6. bombela/backward-cpp - 实现了 C++ 的堆栈回溯便于调试

7. google/googletest - 谷歌单元测试框架

8. google/benchmark - 谷歌性能评估框架

9. glfw/glfw - OpenGL 窗口和上下文管理

10. ibigl/libigl - 各种图形学算法大合集

## Git

### Git是什么

Git是一个版本控制工具, 它可以让你在程序的各个版本之间方便的切换

Github则是Git的线上版本, 可以通过Git将代码快速上传到Github上或下载Github上的代码到本地

### Git基本概念

工作区：仓库的目录。工作区是独立于各个分支的。
暂存区：数据暂时存放的区域，类似于工作区写入版本库前的缓存区。暂存区是独立于各个分支的。
版本库：存放所有已经提交到本地仓库的代码版本
版本结构：树结构，树中每个节点代表一个代码版本。

### Git常用命令

* 设置全局用户名，信息记录在~/.gitconfig文件中

```bash
git config --global user.name xxx
```

* 设置全局邮箱地址，信息记录在~/.gitconfig文件中

```bash
git config --global user.email xxx@xxx.com
```

* 在Git中cd至想要进行版本控制的目录内之后

```bash
git init
```

初始化Git版本控制, 将会生成隐藏的.git文件夹

* 查看仓库状态

```bash
git status
```

* 将XX文件添加到暂存区

```bash
git add xx
```

将目录下所有变化的文件添加到暂存区

```bash
git add .
```

可以使用通配符,如添加所有变化的cpp文件到暂存区

```bash
git add *.cpp
```

* 将文件从仓库索引目录中删掉(不再进行版本管理)

```bash
git rm --cached XX
```

* 将XX文件尚未加入暂存区的修改全部撤销(仍然进行版本管理)

```bash
git restore XX
```

* 将暂存区的内容提交到当前分支

```bash
git commit -m "给自己看的备注信息"
```

* 查看XX文件修改了哪些内容

```bash
git diff XX
```

* 查看当前分支的所有版本

```bash
git log
```

* 查看HEAD指针的移动历史（包括被回滚的版本）

*HEAD指向的节点就是当前所在的版本, 每次commit就会生成一个新节点,同时HEAD也会跟着后移一位*

```bash
git reflog
```

* 将代码库回滚到上一个版本

```bash
git reset --hard HEAD~
```

将代码库回滚100个版本

```bash
git reset --hard HEAD~100
```

将代码库回滚到特定版本

```bash
git reset --hard 版本号
```

其中版本号可以在log中查看(前七个字符--- 7f35f32)

![image-20220302131858230](D:\notebooks\img\image-20220302131858230.png)

也可以在github中查看![image-20220302132002543](.\img\image-20220302132002543.png)

* 将本地仓库关联到远程仓库(得是你自己账号下的仓库)

```bash
git remote add origin git@github.com:XXX/XXX.git
```

这时需要输入设置的邮箱对应账号的密码, 可以配置免密登录, 自行了解sshkeygen

* 将当前分支推送到远程仓库 (第一次需要-u, 以后就不需要了)

```bash
git push -u
```

* 将本地的某个分支推送到远程仓库

```bash
git push origin branch_name
```

* 将远程仓库XXX下载到当前目录下

```
git clone git@github.com:XXX/XXX.git
```

* 将远程仓库的当前分支与本地仓库的当前分支合并

```bash
git pull
```

其他的branch相关命令如果不是多人协作的话是用不上的, 自行了解即可
如果是简单使用的话, 无脑pull-修改-add-commit-push就好了
