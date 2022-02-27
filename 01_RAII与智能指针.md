## RAII与智能指针

#### C++14的lambda允许用auto自动推断类型

```c++
	std::vector<int> v = {4, 3, 2, 1};

    int sum = 0;
    std::for_each(v.begin(),v.end(),[&] (auto vi){
        sum += vi;
    });//模板化

    std::cout << sum << std::endl;

```

#### 面向对象的封装性

如vector类

```c
int main() {
    size_t nv = 2;
    int *v = (int *)malloc(nv * sizeof(int));
    v[0] = 4;
    nv = 4;
    v = (int *)realloc(v, nv * sizeof(int));
    v[2] = 2;
    int sum = 0;
    for (size_t i = 0; i < nv; i++) {
        sum += v[i];
    }
    printf("%d\n", sum);
    free(v);
    return 0;
}//c语言版本
```

```cpp
int main() {
    std::vector<int> v(2);
    v[1] = 3;
    v.resize(4);
    v[2] = 2;
    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    std::cout << sum << std::endl;
    return 0;
}//cpp版本
```

相较于c语言版本,cpp将size和data与各操作函数打包在vector类中,避免了繁琐的修改和手动内存释放

#### 尽量在构造函数中使用初始化表达式

为什么需要初始化表达式？

1. 假如类成员为 const 和引用

2. 假如类成员没有无参构造函数

3. 避免重复初始化，更高效

#### explicit的使用

```cpp
struct Pig {
    std::string m_name;
    int m_weight;
    /*explicit*/ Pig(int weight)
        : m_name("一只重达" + std::to_string(weight) + "kg的猪")
        , m_weight(weight)
    {}
};
int main() {
    Pig pig = 80;//陷阱!
    std::cout << "name: " << pig.m_name << std::endl;
    std::cout << "weight: " << pig.m_weight << std::endl;
    return 0;
}

```

如果不想把pig当成一个整型数,可以在构造函数前加一个explicit,这样就可以避免隐式构造

多个参数时,explicit体现在禁止通过{}来构造对象,如果想要使用return {"佩奇",80};**(非常方便)**之类的返回方式,就不要使用explicit

#### 类型转换:{}与()

{}不允许narrow cast(变窄转换),任何将精度变低的转换是不被允许的

```cpp
int a(3.14);//编译通过
int b{3};//编译通过
int c{3.14};//编译失败

//对于以上的Pig类
Pig pig("佩奇",3.14f);//ok!
Pig pig{"佩奇",3.14f};//不被允许
```

建议在强制类型转换时使用

1. static_cast<int>(3.14f)而不是int(3.14f)
2. reinterpret_cast<void *>(0xb8000)而不是(void *)0xb8000

#### 关于默认构造/拷贝函数

在有自定义构造函数时,如果还要生成默认构造函数,使用以下语句

```cpp
Pig() = default;
Pig(const Pig &) = delete; //删除默认拷贝构造函数
```

#### 从自制vector类说开去

```cpp
struct Vector {
    size_t m_size;
    int *m_data;
    Vector(size_t n) {    // 构造函数 - 对象初始化时调用
        m_size = n;
        m_data = (int *)malloc(n * sizeof(int));
    }
    ~Vector() {           // 解构函数 - 对象销毁时调用
        free(m_data);
    }
    size_t size() {
        return m_size;
    }
    void resize(size_t size) {
        m_size = size;
        m_data = (int *)realloc(m_data, m_size);
    }
    int &operator[](size_t index) {  // 当 v[index] 时调用
        return m_data[index];
    }
};
int main() {
    Vector v(2);
    //构造函数没有explicit,就连Vector v = 2;这种反人类语句都是允许的
    v[0] = 4;
    v[1] = 3;
    v.resize(4);
    v[2] = 2;
    v[3] = 1;
    int sum = 0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    std::cout << sum << std::endl;
}

```

**三五法则**

1. 如果一个类定义了**解构函数**，那么您必须同时定义或删除**拷贝构造函数**和**拷贝赋值函数**，否则出错。

   *拷贝时指针的复制会导致解构时同一块内存会被解构两次或访问无效数据*

   ```cpp
   //合格的拷贝构造函数
   Vector(Vector const &other) {
           m_size = other.m_size;
           m_data = (int *)malloc(m_size * sizeof(int));
           memcpy(m_data, other.m_data, m_size * sizeof(int));
   }
   //合格的拷贝赋值函数
   Vector &operator=(Vector const &other) {
           m_size = other.m_size;
           m_data = (int *)realloc(m_data, m_size * sizeof(int));
           memcpy(m_data, other.m_data, m_size * sizeof(int));
       return *this;//支持连等
   }
   ```

   *拷贝构造函数和拷贝赋值函数的区别*

   ```cpp
   int x = 1;//拷贝构造
   x = 2;//拷贝赋值,先销毁现有数据,再原地拷贝构造新的
   ```

   

2. 如果一个类定义了**拷贝构造函数**，那么您必须同时定义或删除**拷贝赋值函数**，否则出错，删除可导致低效。

3. 如果一个类定义了**移动构造函数**，那么您必须同时定义或删除**移动赋值函数**，否则出错，删除可导致低效。

   *默认的移动构造/赋值函数基本上等于 拷贝构造/赋值 +解构*

   *如果已经自定义移动赋值,为了省力可以删除拷贝赋值,如v2 = v1;时,因为拷贝赋值被删除,编译器会尝试v2 = Vector(v1),从而触发移动赋值*

   

4. 如果一个类定义了**拷贝构造函数**或**拷贝赋值函数**，那么您必须最好同时定义**移动构造函数**或**移动赋值函数**，否则低效。

**新概念swap**

```cpp
template<typename T>
void swap(T a, T b){
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}
```

```cpp
return std::as_const(v2);//显式的拷贝
return std::move(v2);//显式的移动
```

#### RAII解决内存管理的问题: unique_ptr

```cpp
std::unique_ptr<T> p = std::make_unique<>();//相当于new
p = nullptr;//自动解构!
```

**unique_ptr删除了拷贝构造函数**

这导致通常的函数传参无法使用,而这是为了防止之前提到的重复释放的问题

解决方案:

1. 如果func()并不需要控制资源的占有权,并没有掌握对象的生命周期

   通过p.get()获得原始指针,通过func(T *p)的方式传参

2. 如果func()需要占有使用权

   使用func(std::move(p)); 移交控制权,调用func(std::unique_ptr< T> P)

​		**新问题**:如果移交控制权之后还想访问p指向的对象

​		在移交之前,使用p.get()获得原始指针

#### 更智能的指针~~(S宝)~~ shared_ptr

内置计数器,复制加一,解构减一,计数器为零自动释放

**问题**:

1. 维护计数器,有性能上的损失
2. 会导致循环引用,导致在一定情况下一部分内存永远不会被释放

问题二的解决方法 弱引用weak_ptr,弱引用的拷贝和析构不会影响引用计数器

```cpp
std::shared_ptr<T> p = std::make_shared<T>();//引用计数初始化为1
std::weak_ptr<T> weak_p = p;//创建不影响计数器的弱引用
std::cout << weak_p.expired() << std::endl;//expired(),返回weak_p的引用是否失效,由于p引用计数大于0,故未失效,打印0
weak_p.lock()->do_something(); //lock()获得访问p指向对象的权限
```

#### 不同类型智能指针的选择

1.unique_ptr：**当该对象仅仅属于我时**。比如：父窗口中指向子窗口的指针。

2.原始指针：**当该对象不属于我，但他释放前我必然被释放时**。有一定风险。比如：子窗口中指向父窗口的指针。

3.shared_ptr：**当该对象由多个对象共享时，或虽然该对象仅仅属于我，但有使用 weak_ptr 的需要**。

4.weak_ptr：**当该对象不属于我，且他释放后我仍可能不被释放时**。比如：指向窗口中上一次被点击的元素。

5.初学者可以多用 shared_ptr 和 weak_ptr 的组合，更安全。

#### 什么时候考虑三五法则

•一般来说，可以认为符合三五法则的类型是**安全**的。

```cpp
•以下类型是安全的：
•int id; // 基础类型
•std::vector<int> arr; // STL 容器
•std::shared_ptr<Object> child; // 智能指针
•Object *parent; // 原始指针，如果是从智能指针里 .get() 出来的

•以下对象是不安全的：
•char *ptr; // 原始指针，如果是通过 malloc/free 或 new/delete 分配的
•GLint tex; // 是基础类型 int，但是对应着某种资源
std::vector<Object *> objs; // STL 容器，但存了不安全的对象
```

**成员都是安全的类型：五大函数，一个也不用声明** 如果用到自定义解构,说明并不安全,有以下两类

**1. 管理着资源：删除拷贝函数，然后统一用智能指针管理**

**2. 是数据结构：如果可以，定义拷贝和移动**

#### 什么时候考虑常引用

```cpp
//如果是基础类型（比如 int，float）则按值传递：
float squareRoot(float val);
//如果是原始指针（比如 int *，Object *）则按值传递：
void doSomethingWith(Object *ptr);
//如果是数据容器类型（比如 vector，string）则按常引用传递：
int sumArray(std::vector<int> const &arr);
//如果数据容器不大（比如 tuple<int, int>），则其实可以按值传递：
glm::vec3 calculateGravityAt(glm::vec3 pos);
//如果是智能指针（比如 shared_ptr），且需要生命周期控制权，则按值传递：
void addObject(std::shared_ptr<Object> obj);
//如果是智能指针，但不需要生命周期，则通过 .get() 获取原始指针后，按值传递：
void modifyObject(Object *obj);
```

#### *扩展阅读*

1. P-IMPL 模式

2. 虚函数与纯虚函数

3. 拷贝如何作为虚函数

4. std::unique_ptr::release()

5. std::enable_shared_from_this

6. dynamic_cast

7. std::dynamic_pointer_cast

8. 运算符重载

9. 右值引用 &&

10. std::shared_ptr\<void>和 std::any