# 并行常用框架  Intel TBB和OpenMP

## 从并发到并行

### 并发和并行的区别

* 并发：单核处理器，操作系统通过时间片调度算法，轮换着执行着不同的线程，看起来就好像是同时运行一样，其实每一时刻只有一个线程在运行。目的：异步地处理**多个不同的任务**，避免同步造成的**阻塞**。

* 并行：多核处理器，每个处理器执行一个线程，真正的同时运行。目的：将**一个任务**分派到多个核上，从而**更快**完成任务。

  ![image-20220226190604553](.\img\CNcsDI3Z7uEPjk4.png)

### TBB : 任务组

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

