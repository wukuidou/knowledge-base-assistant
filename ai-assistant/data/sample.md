\# Python 基础教程



&#x20;  ## 变量与数据类型

&#x20;  Python 是动态类型语言。常见数据类型包括：

&#x20;  - int：整数，如 1, 100, -5

&#x20;  - float：浮点数，如 3.14, -0.5

&#x20;  - str：字符串，用引号包裹

&#x20;  - list：列表，用 \[] 表示，可修改

&#x20;  - dict：字典，键值对，用 {} 表示



&#x20;  ## 函数

&#x20;  使用 def 关键字定义函数：

&#x20;  ```python

&#x20;  def greet(name):

&#x20;      return f"Hello, {name}!"

&#x20;  ```



&#x20;  ## 列表推导式

&#x20;  一种简洁的创建列表方式：

&#x20;  ```python

&#x20;  squares = \[x\*\*2 for x in range(10)]



