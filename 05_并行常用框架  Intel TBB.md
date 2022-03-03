# 并行常用框架  Intel TBB

## 从并发到并行

### 并发和并行的区别

* 并发：单核处理器，操作系统通过时间片调度算法，轮换着执行着不同的线程，看起来就好像是同时运行一样，其实每一时刻只有一个线程在运行。目的：异步地处理**多个不同的任务**，避免同步造成的**阻塞**。

* 并行：多核处理器，每个处理器执行一个线程，真正的同时运行。目的：将**一个任务**分派到多个核上，从而**更快**完成任务。

  ![image-20220226190604553](.\img\CNcsDI3Z7uEPjk4.png)

### TBB : 任务组

### Intro

如何启用tbb

安装后在CMakelist.txt中添加

```cmake
find_package(TBB REQUIRED)
target_link_libraries(main PUBLIC TBB::tbb)
```

即可

上一篇我们基于标准库进行了多线程运行

```cpp
#include <iostream>
#include <thread>
#include <string>

void download(std::string file) {
    for (int i = 0; i < 10; i++) {
        std::cout << "Downloading " << file
                  << " (" << i * 10 << "%)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

int main() {
    std::thread t1([&] {
        download("hello.zip");
    });
    interact();
    std::cout << "Waiting for child thread..." << std::endl;
    t1.join();
    std::cout << "Child thread exited!" << std::endl;
    return 0;
}
```

如果使用基于TBB的版本,就应该写为

```cpp
#include <iostream>
#include <tbb/task_group.h>
#include <string>

void download(std::string file) {
    for (int i = 0; i < 10; i++) {
        std::cout << "Downloading " << file
                  << " (" << i * 10 << "%)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    std::cout << "Download complete: " << file << std::endl;
}

void interact() {
    std::string name;
    std::cin >> name;
    std::cout << "Hi, " << name << std::endl;
}

int main() {
    tbb::task_group tg;
    tg.run([&] {
        download("hello.zip");
    });
    tg.run([&] {
        interact();
    });
    tg.wait();
    return 0;
}
```

用一个**任务组** tbb::task_group 启动多个任务，一个负责下载，一个负责和用户交互。并在主线程中等待该任务组里的任务全部执行完毕。

区别在于，一个任务不一定对应一个线程，如果任务数量超过CPU最大的线程数，会由 TBB 在用户层负责**调度任务**运行在多个预先分配好的**线程**，而不是由操作系统负责**调度线程**运行在多个物理**核心**。

### 封装版本 : parallel_invoke

只需要依把运行的子函数的lambda版本作为参数即可

以下为并行版本的查找

```cpp
int main() {
    std::string s = "Hello, world!";
    char ch = 'd';
    tbb::parallel_invoke([&] {
        for (size_t i = 0; i < s.size() / 2; i++) {
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    }, [&] {
        for (size_t i = s.size() / 2; i < s.size(); i++) {
            if (s[i] == ch)
                std::cout << "found!" << std::endl;
        }
    });
    return 0;
}

```

## 并行循环

### 并行复杂度

对于并行算法，复杂度的评估则要分为两种：

* 时间复杂度：程序所用的总时间（重点）

* 工作复杂度：程序所用的计算量（次要）

这两个指标都是越低越好。**时间复杂度决定了快慢，工作复杂度决定了耗电量**。

通常来说，工作复杂度 = 时间复杂度 * 核心数量

1个核心工作一小时，4个核心工作一小时。时间复杂度一样，而后者工作复杂度更高。

1个核心工作一小时，4个核心工作1/4小时。工作复杂度一样，而后者时间复杂度更低。

并行的主要目的是**降低时间复杂度**，工作复杂度通常是不变的。甚至有**牺牲工作复杂度换取时间复杂度**的情形。

在上面的例子中,工作复杂度不变,但是时间复杂度变为原来的一半

### 案例 : 并行映射

如果我们需要把1到1<<26范围内的元素i映射为sin(i)并存入数组,朴素的做法显然是这样的

```cpp
int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    for (size_t i = 0; i < n; i++) {
        a[i] = std::sin(i);
    }

    return 0;
}
```

对于一台四核处理器的电脑,它允许我们把这些数字分成四组同时存入数组,这样我们所用的时间就是原来的1/4.

```cpp
int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    size_t maxt = 4;								//线程数
    tbb::task_group tg;
    for (size_t t = 0; t < maxt; t++) {
        auto beg = t * n / maxt;					
        auto end = std::min(n, (t + 1) * n / maxt);	//将数字们分为四份
        tg.run([&, beg, end] {
            for (size_t i = beg; i < end; i++) {
                a[i] = std::sin(i);
            }
        });											//并行
    }
    tg.wait();

    return 0;
}
```

### parallel_for

对于这样的分割循环的操作 , tbb已经封装好了parallel_for函数

```cpp
void parallel_for( const Range &range, const Body &body)
```

```cpp
int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            a[i] = std::sin(i);
        }
    });

    return 0;
}
```

函数会根据设备情况自动分割区间并行运算

* *看上去太复杂了?tbb提供了面向初学者的接口*

```cpp
tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
    a[i] = std::sin(i);
});
```

*这样就简单多了,当然,会有性能上的损失*

PS : 这不就相当于

```cpp
for (int i = 0;i < n ;i++) {
	a[i] = std::sin(i);
}
```

吗

* *一个同样简单的迭代器区间版本*

  ```cpp
  std::vector<float> a(n);
  
  tbb::parallel_for_each(a.begin(), a.end(), [&] (float &f) {
      f = 32.f;
  });
  ```

  *这样就可以把a中的每一个元素设置为32.f*

### 二维区间区间上的for循环 : blocked_range2d

```cpp
blocked_range2d<typename RowValue, typename ColValue>( RowValue row_begin, RowValue row_end, ColValue col_begin, ColValue col_end);
```

```cpp
tbb::parallel_for(tbb::blocked_range2d<size_t>(0,n,0,n), [&](tbb::blocked_range2d<size_t> r){
	for(size_t i = r.cols().begin();i < r.cols().end();i++){
		for(size_t j = r.rows().begin(); j < r.rows().end();j++){
			a[i * n + j] = std::sin(i) * std::sin(j);
		}
	}
});
```

同理还有三维,n维,各个维数对应的称呼如图

![image-20220227160723267](.\img\image-20220227160723267.png)

## 缩并与扫描

### 缩并 (reduce)

什么是缩并捏?

* 加法缩并 : 连加
* 乘法缩并 : 连乘

```cpp
size_t n = 1<<26;
float res = 0;

for (size_t i = 0; i < n; i++) {
    res += std::sin(i);
}
```

这里我们把nn分为c份,每个线程处理n / c的数据之和,最后把所有数据串行相加 时间复杂度为O(n / c + c),而原先的复杂度为O (n^2^)

```cpp
int main() {
    size_t n = 1<<26;
    float res = 0;

	size_t maxt = 4;
    tbb::task_group tg;
    std::vector<float> tmp_res(maxt);
    for(int t = 0;t < maxt; t++) {
        size_t beg = t / maxt * n;
        size_t end = std::min((t + 1) / maxt * n,n);
        tg.run([&, t, beg, end]{
            float local_res = 0;
           	for(size_t i = beg;i < end;i++){
               local_res += std::sin(i);
           	} 
            tmp_res.push_back(local_res);
        });
    }
	tg.wait();
    for(size_t t = 0; t < maxt;t++) {
        res += tmp_res[t];
    }
    
    std::cout << res << std::endl;
    return 0;
}
```

原理 : 加法交换律 + 加法结合律

当然,更优化的选择是反复递归,这样时间复杂度可以被优化为O(logn),而工作复杂度几乎不变

缩并在一定程度上可以减少浮点误差

#### parallel_reduce

当然,tbb也提供了缩并的封装版本parallel_reduce

```cpp
Value parallel_reduce(const Range &range, const Value &identity, const RealBody &real_body, const Reduction &reduction)	//区间  初始值  给定区间之后的操作  区间结果之间的操作
```

```cpp
#include <iostream>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    float res = tbb::parallel_deterministic_reduce(tbb::blocked_range<size_t>(0, n), (float)0,
    [&] (tbb::blocked_range<size_t> r, float local_res) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += std::sin(i);
        }
        return local_res;
    }, [] (float x, float y) {
        return x + y;
    });

    std::cout << res << std::endl;
    return 0;
}
```

#### 保证运算结果在浮点上也一致 : parallel_deterministic_reduce

用法和上面一样

### 扫描 (scan)

如图所示，扫描和缩并差不多，只不过他会把求和的**中间结果存到数组里去**,也就是计算前缀和

![image-20220227163951088](.\img\image-20220227163951088.png)

```cpp
int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    float res = 0;

    for (size_t i = 0; i < n; i++) {
        res += std::sin(i);
        a[i] = res;
    }

    std::cout << a[n / 2] << std::endl;
    std::cout << res << std::endl;
    return 0;
}
```

### parallel_scan

```cpp
#include <iostream>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <cmath>

int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);
    float res = tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (float)0,
    [&] (tbb::blocked_range<size_t> r, float local_res, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_res += std::sin(i);
            if (is_final) {
                a[i] = local_res;
            }
        }
        return local_res;
    }, [] (float x, float y) {
        return x + y;
    });

    std::cout << a[n / 2] << std::endl;
    std::cout << res << std::endl;
    return 0;
}
```

*不大懂()*

## 性能测试

### tbb提供的测试 : tbb::tick_count::now();

等于std::chrono::steady_clock::now(),但是是由tbb提供的,在tbb框架下更精确

### 加速比

加速比=串行用时÷并行用时 , 理想加速比应该是核心的数量。

在 6 个物理核心，12 个逻辑核心的电脑测试得到 : 

* for 部分加速比为 **5.98** 倍。

* reduce 部分加速比为 **10.36** 倍。

似乎这里 reduce 的加速比是逻辑核心数量，而 for 的加速比是物理核心的数量？

因为本例中 reduce 是内存密集型，for 是计算密集型。

超线程对 reduce 这种只用了简单的加法，瓶颈在内存的算法起了作用。

而本例中 for 部分用了 std::sin，需要做大量数学运算，因此瓶颈在 ALU

### 性能测试框架 : Google benchmark

手动计算时间差有点太硬核了，而且只运行一次的结果可能不准确，最好是多次运行取平均值才行

因此可以利用谷歌提供的这个框架。

只需将你要测试的代码放在他的for (auto _: bm)里面即可。他会自动决定要重复多少次，保证结果是准确的，同时不浪费太多时间。

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <benchmark/benchmark.h>

constexpr size_t n = 1<<27;
std::vector<float> a(n);

void BM_for(benchmark::State &bm) {
    for (auto _: bm) {
        // fill a with sin(i)
        for (size_t i = 0; i < a.size(); i++) {
            a[i] = std::sin(i);
        }
    }
}
BENCHMARK(BM_for);

void BM_reduce(benchmark::State &bm) {
    for (auto _: bm) {
        // calculate sum of a
        float res = 0;
        for (size_t i = 0; i < a.size(); i++) {
            res += a[i];
        }
        benchmark::DoNotOptimize(res); // 防止编译器发现res没有被打印将其自动优化掉
    }
}
BENCHMARK(BM_reduce);

BENCHMARK_MAIN();
```

使用方法 : 将benchmark文件夹放在代码目录下， 在CMakeist.txt里引入package

```cmake
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off the fking test!")
add_subdirectory(benchmark)
target_link_libraries(main PUBLIC benchmark)
```

## 任务域和嵌套

### 任务域 : tbb:task_arena

使用

```cpp
int main() {
    size_t n = 1<<26;
    std::vector<float> a(n);

    tbb::task_arena ta;
    ta.execute([&] {
        tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
            a[i] = std::sin(i);
        });
    });

    return 0;
}
```

表明for循环在ta任务域中进行

### 任务域 : 指定使用4个线程

我们可以在定义任务域时规定有多少个线程用来执行任务域内的循环

```cpp
    tbb::task_arena ta(4);	//4个线程
    ta.execute([&] {
        tbb::parallel_for((size_t)0, (size_t)n, [&] (size_t i) {
            a[i] = std::sin(i);
        });
    });
```

 ### for循环的嵌套

与普通的for循环相同 parallel_for也可以嵌套运行

```cpp
int main() {
	size_t n = 1 << 13;
	std::vector<float> a(n * n);
	
	tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
		tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
			a[i * n + j] = std::sin(i) * std::sin(j);
		});
	});
	return 0;
}
```

### 嵌套for循环的死锁问题

```cpp
int main() {
	size_t n = 1 << 13;
	std::vector<float> a(n * n);
	std::mutex mtx; 
	
	tbb::parallel_for((size_t)0, (size_t)n, [&](size_t i){
		std::lock_guard lck(mtx);
		tbb::parallel_for((size_t)0, (size_t)n, [&](size_t j){
			a[i * n + j] = std::sin(i) * std::sin(j);
		});
	});
	return 0;
}
```

为什么这样会导致死锁呢?

因为 TBB 用了**工作窃取法**来分配任务：当一个线程 t1 做完自己队列里全部的工作时，会从另一个工作中线程 t2 的队列里取出任务，以免 t1 闲置浪费时间。

因此内部 for 循环有可能“窃取”到另一个外部 for 循环的任务，从而导致 mutex 被重复上锁。

#### 解决方案1 : 使用标准库的递归锁 std : recursive_lock

```cpp
std::recursive_lock mtx; 
```

#### 解决方案2 : 创建另一个任务域,这样不同域之间就不会窃取工作

```cpp
int main(){
	size_t n = 1 << 13;
	vector<float> a(n * n);
	std::mutex mtx;
	
	tbb::parallel_for((size_t)0, (size_t)n,[&](size_t i){
		std::lock_guard lck(mtx);
		tbb::task_arena	ta;
		ta.execute([&]{
			tbb::parallel_for((size_t)0, (size_t) n, [&](size_t j){
				a[i * n + j] = std::sin(i) * std::sin(j);
			});
		});
	});
	
	return 0;
}
```

#### 解决方案3 : 同一个任务域,但是用isolate隔离,禁止其内部工作被窃取(推荐)

```cpp
tbb::parallel_for((size_t)0, (size_t)n,[&](size_t i){
	std::lock_guard lck(mtx);
	tbb::this_task_arena::isolate([&]{
		tbb::parallel_for((size_t)0, (size_t) n, [&](size_t j){
			a[i * n + j] = std::sin(i) * std::sin(j);
		});
	});
});
```

## 任务分配

对于并行计算，通常都是 CPU 有几个核心就开几个线程，因为我们只要同时执行就行了嘛。

比如 cornell box 这个例子里，我们把图片均匀等分为四块处理。然而发现4号线程所在的块，由于在犄角旮旯里光线反弹的次数多，算得比其他块的慢，而有的块却算得快。但是因为木桶原理，最后花的时间由最慢的那个线程决定，因此变成1分30秒了，多出来的30秒里1号和2号核心在闲置着，因为任务简单已经算完了，只有4号核心一个人在处理额外的光线。

![image-20220227201352465](.\img\image-20220227201352465.png)

### 解决方案1 : 线程数量超过**CPU核心数量，让系统调度保证各个核心始终饱和**

因此，最好不是按照图像大小均匀等分，而是按照工作量大小均匀等分。然而工作量大小我们没办法提前知道……怎么办？

最简单的办法：只需要让线程数量超过CPU核心数量，这时操作系统会自动启用时间片轮换调度，轮流执行每个线程。

比如这里分配了16个线程，但是只有4个处理器核心。那么就会先执行1,2,3,4号线程，一段时间后自动切换到5,6,7,8线程。当一个线程退出时候，系统就不会再调度到他上去了，从而保证每个核心始终有事可做。

![image-20220227201531983](.\img\image-20220227201531983.png)

问题 : 可能导致一个核心一会儿处理这个,一会儿处理那个,导致缓存局部性被破坏

### 解决方案 2 : 线程数量不变，但是用一个队列分发和认领任务

所以另一种解法是：我们仍是分配4个线程，但还是把图像切分为16份，作为一个“任务”推送到全局队列里去。每个线程空闲时会不断地从那个队列里取出数据，即“认领任务”。然后执行，执行完毕后才去认领下一个任务，从而即使每个任务工作量不一也能自动适应。

这种技术又称为线程池（thread pool），避免了线程需要保存上下文的开销。但是需要我们管理一个任务队列，而且要是线程安全的队列。

### 解决方案3 : 每个线程一个任务队列，做完本职工作后可以认领其他线程的任务

![image-20220227201927965](.\img\image-20220227201927965.png)

### tbb::static_partitioner

将数据分为数个任务,且作为parallel_for的参数存在,如

```cpp
int main() {
	size_t n = 32;
	
	tbb::task_arena ta(4);
	ta.execute([&]{
		tbb::parallel_for(tbb::bloacked_range<size_t>(0,n),
        [&](tbb::blocked_range<size_t> r){
			//do something
		}, tbb::static_partitioner{});
	});
	
	return 0;
}
```

其中tbb:task_arena分出了四个线程,而tbb:static_partitioner将32的数据分为4个任务,每个任务有8个元素

此外,我们可以在tbb::blocked_range中添加第三个参数指定数据的粒度,如

```cpp
int main() {
size_t n = 32;
	
tbb::task_arena ta(4);
ta.execute([&]{
	tbb::parallel_for(tbb::bloacked_range<size_t>(0,n,16), 
    [&](tbb::blocked_range<size_t> r){
		//do something
	}, tbb::static_partitioner{});
});
	
return 0;
```

这时,tbb::static_partitioner将数据分成了各有16个元素的任务

tbb::static_partitioner对于循环体不均匀的情况效果不好

### tbb::simple_partitioner

默认按最小粒度分开,比如tbb::blocked_range的默认粒度是1,那么tbb::simple_partitioner会把数据分为32个任务

如果我们在tbb:blocked_range中把粒度设置为4,那么将创建8个任务

tbb::simple_partitioner对于循环体不均匀的情况效果很好

### tbb::auto_partitioner (默认)

自动根据lambda中函数的执行时间判断采取何种分配方法

此外,tbb::this_task_arena::max_concurrency能够返回当前参与运行的线程数

### 案例 : 矩阵转置

对于1 << 14 的数据量 , 当我们把粒度设置为16时，simple_partitioner 比 auto_partitioner 快 **3.31**倍

为什么捏

tbb::simple_partitioner 能够按照给定的粒度大小（grain）将矩阵进行分块。块内部小区域按照常规的两层循环访问以便矢量化，块外部大区域则以类似 **Z** **字型**的曲线遍历，这样能保证每次访问的数据在地址上比较靠近，并且都是最近访问过的，从而已经在缓存里可以直接读写，避免了从主内存读写的超高延迟。



![image-20220227204936956](.\img\image-20220227204936956.png)

## 并发容器

### std::vector 扩容时会移动元素

std::vector 内部存储了一个指针，指向一段**容量** **capacity** 大于等于其 size 的内存。

众所周知，push_back 会导致 size 加 1，但当他看到容量 capacity 等于当前 size 时，意识到无法再追加新数据。这时他会重新 malloc 一段更大的连续内存，使得 capacity 变成 size 的**两倍**，并且把旧的数据**移动**过去，然后继续追加数据。

这就导致**前半段的元素的地址被改变**，从而导致之前保存的**指针和迭代器失效**。

如果预先知道size的最后会是n,则可以调用reserve(n)**预分配**一段大小为n的内存,从而capacity一开始就等于n.这样push_back就不需要动态扩容,从而避免了元素被移动导致指针和迭代器失效

### 不连续的 tbb::concurrent_vector

std::vector 造成指针失效的根本原因在于他必须保证内存是**连续的**，从而不得不在扩容时移动元素。

因此可以用 tbb::concurrent_vector，他不保证元素在内存中是连续的。换来的优点是 push_back 进去的元素，扩容时不需要移动位置，从而**指针和迭代器不会失效**。

同时他的 push_back 会额外返回一个迭代器（iterator），指向刚刚插入的对象。

#### grow_by 一次性扩容一定大小

push_back 一次只能推入一个元素。

而 grow_by(n) 则可以一次扩充 n 个元素。他同样是返回一个迭代器（iterator），之后可以通过迭代器的 **++运算符**依次访问连续的 n 个元素，* **运算符**访问当前指向的元素。

```cpp
int main() {
    size_t n = 1<<10;
    tbb::concurrent_vector<float> a;

    for (size_t i = 0; i < n; i++) {
        auto it = a.grow_by(2);
        *it++ = std::cos(i);
        *it++ = std::sin(i);
    }

    std::cout << a.size() << std::endl;

    return 0;
}
```

#### tbb::concurrent_vector的并发性

除了内存不连续、指针和迭代器不失效的特点，tbb::concurrent_vector 还是一个**多线程安全**的容器，能够被多个线程同时并发地 grow_by 或 push_back 而不出错。

而 std::vector 只有只读的 .size() 和 [] 运算符是安全的，且不能和写入的 push_back 等一起用，否则需要用读写锁保护

#### 不建议通过索引访问

因为 tbb::concurrent_vector **内存不连续**的特点，通过索引访问，比通过迭代器访问的效率低一些。

因此不推荐像 a[i] 这样通过索引随机访问其中的元素，*(it + i) 这样需要迭代器跨步访问的也不推荐。

最好通过迭代器顺序访问

#### parallel_for的迭代器访问

```cpp
int main() {
	size_t n = 1 << 10;
	tbb::concurrent_vector<float> a(n);
	
	tbb::parallel_for(tbb::blocked_range(a.begin(), a.end()),
	[&](tbb::blocked_range<decltype(a.begin())> r){
		for(auto it = r.begin(); it != r.end(); ++it) {
			*it += 1.0f; 
		}
	});
}
```

冷知识：tbb::blocked_range 的参数不一定是 size_t，也可以是迭代器表示的区间。

这样 lambda 体内 r 的 begin 和 end 也会返回 tbb::concurrent_vector 的迭代器类型。

第一个 tbb::blocked_range 尖括号里的类型可以省略是因为 C++17 的 CTAD 特性。第二个则是用了 decltype 自动推导，也可以 (auto r)

*很多stl容器都有多线程安全版本*

## 并行筛选

筛选sin(i),将大于0的值存入数组

### 朴素版

```cpp
size_t n = 1<<27;
std::vector<float> a;

for (size_t i = 0; i < n; i++) {
    float val = std::sin(i);
    if (val > 0) {
        a.push_back(val);
    }
}
```

### 并行筛选1

```cpp
size_t n = 1 << 27;
tbb::concurrent<float> a;

tbb::parallel_for(tbb::blaocked_range<size_t>(0,n), \
[&](tbb::blocked_range<size_t> r){
    for(size_t i = r.begin(); i < r.end(); i++) {
        float val = std::sin(i);
        if(val > 0) {
            a.push_back(val);
        }
    }
});	//加速比：1.32 倍
```

利用多线程安全的 concurrent_vector 动态追加数据

基本没有加速,可能是因为为了安全内部使用了大量锁

### 并行筛选2

先推到线程局部（thread-local）的 vector

最后一次性推入到 concurrent_vector

可以避免频繁在 concurrent_vector 上产生锁竞争

```cpp
size_t n = 1 << 27;
tbb::concurrent<float> a;

tbb::parallel_for(tbb::blocked_range<size_t>(0,n), 
[&](tbb::blocked_range<size_t> r){
   	std::vector<float> local_a;
    for (size_t i = r.begin(); i < r.end(); i++) {
        float val = std::sin(i);
        if(val > 0) {
            local_a.push_back(val);
        }
    }
    auto it = a.gorw_by(local_a.size);
    for(size_t = 0;i < local_a.size(); i++) {
        *it++ = local_a[i];
    }
});	//加速比:5.55倍
```

### 并行筛选3

线程局部的 vector 调用 reserve 预先分配一定内存

避免 push_back 反复扩容时的分段式增长

同时利用标准库的 std::copy 模板简化了代码

```cpp
int main() {
    size_t n = 1<<27;
    tbb::concurrent_vector<float> a;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<float> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        auto it = a.grow_by(local_a.size());
        std::copy(local_a.begin(), local_a.end(), it);
    });
}

```

### 并行筛选4

如果需要筛选后的数据是连续的，即 a 是个 std::vector，这时就需要用 mutex 锁定，避免数据竞争

```cpp
int main() {
    size_t n = 1<<27;
    std::vector<float> a;
    std::mutex mtx;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<float> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        std::lock_guard lck(mtx);
        std::copy(local_a.begin(), local_a.end(), std::back_inserter(a));
    });

    return 0;
}
```

### 并行筛选5(推荐)

先对 a 预留一定的内存，避免频繁扩容影响性能

```cpp
int main() {
    size_t n = 1<<27;
    std::vector<float> a;
    std::mutex mtx;

    a.reserve(n * 2 / 3);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<float> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        std::lock_guard lck(mtx);
        std::copy(local_a.begin(), local_a.end(), std::back_inserter(a));
    });

    return 0;
}
```

### 并行筛选6

使用 tbb::spin_mutex 替代 std::mutex。spin_mutex（基于硬件原子指令）会让 CPU 陷入循环等待，而不像 mutex（操作系统提供调度）会让线程进入休眠状态的等待。

若**上锁的区域较小**，可以用轻量级的 spin_mutex。若上锁的区域很大，则循环等待只会浪费 CPU 时间。这里锁的区域是 std::copy，比较大，所以 spin_mutex 效果不如 mutex 好

```cpp
int main() {
    size_t n = 1<<27;
    std::vector<float> a;
    tbb::spin_mutex mtx;

    a.reserve(n * 2 / 3);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
    [&] (tbb::blocked_range<size_t> r) {
        std::vector<float> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            float val = std::sin(i);
            if (val > 0) {
                local_a.push_back(val);
            }
        }
        std::lock_guard lck(mtx);
        std::copy(local_a.begin(), local_a.end(), std::back_inserter(a));
    });

    return 0;
}
```

### 并行筛选7

线程粒度很细，核心数量很多的 GPU，往往没办法用 concurrent_vector 和 thread-local vector。或是你需要保证**筛选前后顺序不变**。这时要把筛选分为三步：

一、算出每个元素需要往 vector 推送数据的数量（本例中只有 0 和 1 两种可能）

二、对刚刚算出的数据进行并行扫描（scan），得出每个 i 要写入的索引。

三、再次对每个元素并行 for 循环，根据刚刚生成写入的索引，依次写入数据

## 分治和排序

### 从斐波那契数列开始

#### 朴素版

```cpp
int fib(int n) {
    if (n < 2)
        return n;
    int first = fib(n - 1);
    int second = fib(n - 2);
    return first + second;
}

int main() {
    std::cout << fib(39) << std::endl;
    return 0;
}	//1.06s
```

可以发现first和second的计算无关联的,可以并行解决

#### 并行版

```cpp
int fib(int n) {
    if (n < 2)
        return n;
    int first, second;
    tbb::parallel_invoke([&]{
        first = fib(n - 1);
    }, [&] {
        second = fib(n - 2);
    });
    tg.wait();
    return first + second;
}	//2.61s
```

但是反而变慢了,因为分开也是需要时间的

#### 解决方案

任务划分得够细时，转为串行，缓解调度负担（scheduling overhead）

```cpp
int serial_fib(int n) {
    if (n < 2)
        return n;
    int first = serial_fib(n - 1);
    int second = serial_fib(n - 2);
    return first + second;
}

int fib(int n) {
    if (n < 29)
        return serial_fib(n);
    int first, second;
    tbb::parallel_invoke([&] {
        first = fib(n - 1);
    }, [&] {
        second = fib(n - 2);
    });
    return first + second;
}
```

![image-20220227215657370](.\img\image-20220227215657370.png)

### 快速排序(qucik_sort)

我们知道,cpp标准库提供了sort函数

```cpp
int main() {
    size_t n = 1<<24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);
    
    std::sort(arr.begin(), arr.end(), std::less<int>{});
    
    return 0;
}	//1.03273s
```

,与我们手写的版本对比

```cpp
template <class T>
void quick_sort(T *data, size_t size) {
    if (size < 1)
        return;
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;
    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while (left < right) {
        while (left < right && !(data[right] < pivot))
            right--;
        if (left < right)
            data[left++] = data[right];
        while (left < right && data[left] < pivot)
            left++;
        if (left < right)
            data[right--] = data[left];
    }
    data[left] = pivot;
    quick_sort(data, left);
    quick_sort(data + left + 1, size - left - 1);	//这里!可以并行优化!
}

int main() {
    size_t n = 1<<24;
    std::vector<int> arr(n);
    std::generate(arr.begin(), arr.end(), std::rand);
    
    std::quick_sort(arr.begin(), arr.end(), std::less<int>{});
    
    return 0;
}	//1.31s
```

std::hash 用于从输入生成随机数，输入不变则结果不变。

随机枢轴的位置防止数据已经有序造成最坏的 O(n²)

按照注释给出的位置优化之后,加速比达到了2.05倍

```cpp
template <class T>
void quick_sort(T *data, size_t size) {
    if (size < 1)
        return;
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;
    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while (left < right) {
        while (left < right && !(data[right] < pivot))
            right--;
        if (left < right)
            data[left++] = data[right];
        while (left < right && data[left] < pivot)
            left++;
        if (left < right)
            data[right--] = data[left];
    }
    data[left] = pivot;
    tbb::parallel_invoke([&] {
        quick_sort(data, left);
    }, [&] {
        quick_sort(data + left + 1, size - left - 1);
    });
}
```

比标准库还快 !

还能快吗?那自然要利用之前提到的技术,在数据足够小时开始串行排序

```cpp
template <class T>
void quick_sort(T *data, size_t size) {
    if (size < 1)
        return;
    if (size < (1<<16)) {
        std::sort(data, data + size, std::less<T>{});
        return;
    }	//这里!
    size_t mid = std::hash<size_t>{}(size);
    mid ^= std::hash<void *>{}(static_cast<void *>(data));
    mid %= size;
    std::swap(data[0], data[mid]);
    T pivot = data[0];
    size_t left = 0, right = size - 1;
    while (left < right) {
        while (left < right && !(data[right] < pivot))
            right--;
        if (left < right)
            data[left++] = data[right];
        while (left < right && data[left] < pivot)
            left++;
        if (left < right)
            data[right--] = data[left];
    }
    data[left] = pivot;
    tbb::parallel_invoke([&] {
        quick_sort(data, left);
    }, [&] {
        quick_sort(data + left + 1, size - left - 1);
    });
}
```

加速比达到了4.59倍捏;

然而,这种排序已经有人帮你封装好了,tbb::parallel_sort可以达到一样的效果

## 流水线并行

### 案例 : 批量处理数据  

#### 原案

```cpp
struct Data {
    std::vector<float> arr;

    Data() {
        arr.resize(std::rand() % 100 * 500 + 10000);
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = std::rand() * (1.f / (float)RAND_MAX);
        }
    }

    void step1() {
        for (int i = 0; i < arr.size(); i++) {
            arr[i] += 3.14f;
        }
    }

    void step2() {
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size() - 1; i++) {
            tmp[i] = arr[i - 1] + arr[i] + arr[i + 1];
        }
        std::swap(tmp, arr);
    }

    void step3() {
        for (int i = 0; i < arr.size(); i++) {
            arr[i] = std::sqrt(std::abs(arr[i]));
        }
    }

    void step4() {
        std::vector<float> tmp(arr.size());
        for (int i = 1; i < arr.size() - 1; i++) {
            tmp[i] = arr[i - 1] - 2 * arr[i] + arr[i + 1];
        }
        std::swap(tmp, arr);
    }
};

int main() {
    size_t n = 1<<12;

    std::vector<Data> dats(n);

    for (auto &dat: dats) {
        dat.step1();
        dat.step2();
        dat.step3();
        dat.step4();
    }
    return 0;
}
```

显然这里的 for (auto &dat: dats)可以并行计算,试试用tbb::parallel_for?

```cpp
tbb::parallel_for_each(dats.begin(), dats.end(), [&] (Data &dat) {
    dat.step1();
    dat.step2();
    dat.step3();
    dat.step4();
});
```

加速比：**3.16** 倍

很不理想，为什么

很简单，循环体太大，每跑一遍指令缓存和数据缓存都会重新失效一遍。且每个核心都在读写不同地方的数据，不能很好的利用三级缓存，导致内存成为瓶颈



#### 解决方案1 : 拆分为三个for

加速比：**3.47** 倍

解决了指令缓存失效问题，但是三次独立的for循环每次结束都需要同步，一定程度上妨碍了CPU发挥性能；而且每个step后依然写回了数组，数据缓存没法充分利用

```cpp
tbb::parallel_for_each(dats.begin(), dats.end(), [&] (Data &dat) {
    dat.step1();
});
tbb::parallel_for_each(dats.begin(), dats.end(), [&] (Data &dat) {
    dat.step2();
});
tbb::parallel_for_each(dats.begin(), dats.end(), [&] (Data &dat) {
    dat.step3();
});
tbb::parallel_for_each(dats.begin(), dats.end(), [&] (Data &dat) {
    dat.step4();
});
```

#### 解决方案2 : 流水线并行

加速比：**6.73** 倍

反直觉的并行方式，但是加速效果却很理想，为什么？

流水线模式下每个线程都只做自己的那个步骤（filter），从而对指令缓存更友好。且一个核心处理完的数据很快会被另一个核心用上，对三级缓存比较友好，也节省内存。

且 TBB 的流水线，其实比教科书上描述的传统流水线并行更加优化：

他在 t1 线程算完 d1 的 s1 时，会继续让 t1 负责算 d1 的 s2，这样 d1 的数据就是在二级缓存里，比调度到让 t2 算需要进入三级缓存更高效。而当 t2 的队列比较空时，又会让 t1 继续算 d2 的 s2，这样可以避免 t2 闲置浪费时间。总之就是会自动负载均衡非常智能，完全无需操心内部细节。

```cpp
auto it = dats.begin();
tbb::parallel_pipeline(8
, tbb::make_filter<void, Data *>(tbb::filter_mode::serial_in_order,
[&] (tbb::flow_control &fc) -> Data * {
    if (it == dats.end()) {
        fc.stop();
        return nullptr;
    }
    return &*it++;
})
, tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
[&] (Data *dat) -> Data * {
    dat->step1();
    return dat;
})
, tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
[&] (Data *dat) -> Data * {
    dat->step2();
    return dat;
})
, tbb::make_filter<Data *, Data *>(tbb::filter_mode::parallel,
[&] (Data *dat) -> Data * {
    dat->step3();
    return dat;
})
, tbb::make_filter<Data *, void>(tbb::filter_mode::parallel,
[&] (Data *dat) -> void {
    dat->step4();
})
)
```

#### 流水线并行 : filter参数

serial_in_order 表示当前步骤只允许串行执行，且执行的顺序必须一致。

serial_out_of_order 表示只允许串行执行，但是顺序可以打乱。

parallel 表示可以并行执行当前步骤，且顺序可以打乱。

每一个步骤（filter）的输入和返回类型都可以不一样。要求：流水线上一步的返回类型，必须和下一步的输入类型一致。且第一步的没有输入，最后一步没有返回，所以都为 void。

TBB 支持嵌套的并行，因此流水线内部也可以调用 tbb::parallel_for 进一步并行。

#### 流水线并行的特点

流水线式的并行，因为每个线程执行的指令之间往往没有关系，主要适用于各个核心可以独立工作的 CPU，GPU 上则有 stream 作为替代。

流水线额外的好处是可以指定一部分 filter 为串行的（如果他们没办法并行调用的话）而其他 filter 可以和他同时并行运行。这可以应对一些不方便并行，或者执行前后的数据有依赖，但是可以拆分成多个步骤（filter）的复杂业务。

还有好处是他无需先把数据全读到一个内存数组里，可以**流式**处理数据（on-fly），节省内存。

不过需要注意流水线每个步骤（filter）里的工作量最好足够大，否则无法掩盖调度overhead。
