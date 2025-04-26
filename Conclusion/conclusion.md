# NS conclusion

## 解题总结
- **读题**：怎么你最会的到这里就不愿意动了呢？**其实拿到题目第一件事去就是读题，找资料**，看看题目到底要你干什么，以及你需要做什么，不然都是一头雾水，不知道在干什么。像去水印那道题目，你始终不只到水印的mask有什么用，非常后面你才明白了，那时候已经来不及了。当然，很多事情只有你自己调用了一个model之后才能真正的理解。
- **框架和分解**：写model时一定要把框架搭好，**model、utils、preprocess、checkpoint**。这样就把你的问题划分成一个一个子问题，难度就会降低好多。你需要一开始就思考好你的解题顺序，可以用**倒推**的方式，首先需要把**模型训练**好，已经训练好就直接进行下一步;其次就是计算答案。你通过这些结果，反推你需要什么样的数据预处理，以及其他的，等模型跑起来了，这些时间你就可以用来写剩下的其他处理结果的脚本。
- **check**：运行时一定要先把epoch设定为**1**,debug完之后再真正开始跑模型。计算答案时，也先只算**1**个input，例如去水印，就先去一个图片的水印，处理好了，再换for循环！
- **找模型**：尽量找别人现成的模型用，自己训练的实在是太拉垮了，自己根本没得救，又训练的慢。
- **Chaptgpt**: 你在这场比赛中过于依赖`gpt`了，浪费了你的大量时间，很多问题它真的理解不了，很多报错它也答非所问。**那些错误，问题其实你可以先尝试着解决一下**，很多其实也不难，而且这样才能真正地锻炼你的debug能力，不是扔给gpt就万事大吉的。像很多模型的文档，其实你自己也看得懂，只是你不想动你的脑子而已。
- **心态**：这点还是要夸夸自己的，即使要期中考，即使水印写了三天，你的心态依然没炸，最后真的把水印磨出来了;也坚持到了最后，挑起了大梁，一开始我以为我是被带飞的，没想到还**carry**了一下队友，哈哈哈哈哈哈哈。
- **writeup**：以后如果还有这种需要写writeup的比赛，可以把运行的中间结果**截图**下来，不然最后很麻烦。
- **log**: 程序运行时一定要多加**print和assert**，能少出很多bug，以及加深对程序运行的理解。还有运行结果和报错可以写**log**中，log真是一个非常好用的东西！
- **合作**：这场比赛，我们太缺少合作了，大家在写的时候肯定会有非常多的心得和教训的，其实我们应该开个会，然后大家一起讨论和总结，肯定能得出非常多的灵感和启发的。

## 知识点总结

### 处理各类文件
- **with**:文件处理器
  - strip():移除每行前后多余的空格和换行符
  - split:将每行内容切割成一个列表。
```python
with open(rating_file, 'r') as f:
    for line in f:
        u, v, r = line.strip().split()
        u, v = int(u), int(v)
        r = float(r)
        user_ratings[u].append((v, r))

```
- `try and except`:尝试可能出错的代码
```python
try:
    with open(input_json_path, "r") as f:
        qa_list = json.load(f)
except FileNotFoundError:
    print(f"[❌ Error] JSON 文件未找到：{input_json_path}")
    exit(1)  # 如果找不到 JSON 文件，直接退出程序

try:
    # 代码块
    some_code()
except Exception as e:
    print(f"错误: {e}")
finally:
    # 无论如何都会执行的代码
    print("清理工作")

```


#### 文件path
1. os 库
- 路径拼接
```python
import os

# 拼接文件路径
folder = 'folder_name'
filename = 'file.txt'

# 在不同操作系统中自动处理路径分隔符
file_path = os.path.join(folder, filename)
print(file_path)
```
- 创建文件夹
  - `./`:表示当前目录
  - `../`表示上级目录
```python
json_path = "structured.json"
output_path = "./results_xml/results.xml"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"\n🎉 所有问题推理完成，结果已保存至：{output_path}")

```

#### 日志读写
- 其实和读写其他文件差不多
- `w`:写模式，如果文件不存在，则会创建一个新文件。
```python
# 读取日志文件内容
with open('logfile.txt', 'r', encoding='utf-8') as f:
    content = f.read()  # 读取整个文件内容
    print(content)

# 按行读取日志
with open('logfile.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())  # 使用 strip() 去掉每行末尾的换行符



# 写入日志文件（会覆盖已有的日志）
with open('logfile.txt', 'w', encoding='utf-8') as f:
    f.write("这是第一次写入的日志\n")
    f.write("写入的时间是：2025年4月26日\n")

# 追加日志到文件（不会覆盖已有内容）
with open('logfile.txt', 'a', encoding='utf-8') as f:
    f.write("这是追加的日志\n")
    f.write("追加的时间是：2025年4月26日\n")

```

- python中有logging库便于写入日志，这种是记录关键日志的，如果想记录自己的日志，还是自己写吧。

```python
import logging

# 配置日志
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 写入日志
logging.debug("这是调试日志")
logging.info("这是信息日志")
logging.warning("这是警告日志")
logging.error("这是错误日志")
logging.critical("这是严重日志")

logging.basicConfig(filename='app.log', level=logging.DEBUG, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("这是追加的日志")
```
#### csv文件
- excel表格就是csv文件
```csv
姓名,年龄,部门
Alice,29,研发部
Bob,34,市场部
Charlie,25,财务部
```
- python使用csv方法，**with**可以看作一把互斥锁，如果程序中断了，会自动关闭文件且返还。
  - `with open('output.csv', 'w', newline='', encoding='utf-8') as f`：'w':写模型，会把源文件清空，'a'：追加模型，会接着往源文件写入；as f :的意思为吧f当作文件描述符。
  - ` writer.writeheader() `：写入表头，就是写入excel中第一行，姓名等信息
```python
import csv
# 读取 CSV
with open('employees.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # row 是一个字典，如 {'姓名': 'Alice', '年龄': '29', '部门': '研发部'}
        print(row['姓名'], row['部门'])

# 写入 CSV
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['姓名', '年龄', '部门']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    writer.writerow({'姓名': 'David', '年龄': '30', '部门': 'HR部'})
```

- 以下是pandas中使用csv的用法
  - `header = False`:不写入表头，避免重复
  - `idnex = False`:不使用行index
```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('output.csv', encoding='utf-8')

# 打印读取的内容
print(df)

import pandas as pd

# 创建要追加的 DataFrame
new_data = pd.DataFrame({
    '姓名': ['王五'],
    '年龄': [28],
    '部门': ['财务部']
})

# 追加数据到 CSV
new_data.to_csv('output.csv', mode='a', header=False, index=False, encoding='utf-8')

# 选择“年龄大于 25”的行
df = df[df['年龄'] > 25]

# 计算“年龄”列的平均值
avg_age = df['年龄'].mean()

# 删除缺失值
df = df.dropna()
```

#### json 文件
- JSON 文件存储的是键值对（key-value pairs）数据，采用类似于 Python 字典或 JavaScript 对象的格式。它的主要特点是 键（key）和对应的值（value）通过冒号 : 分隔，每对键值对之间用逗号 , 分隔。
- 由于json文件基于文本编写，所以常用于不同系统之间传输数据
- JSON 支持以下基本数据类型：
> 字符串（String）：用双引号包裹，例如 "Alice"。
> 数字（Number）：例如 30 或 3.14。
> 布尔值（Boolean）：true 或 false。
> 空值（null）：表示空值。
> 对象（Object）：由 {} 包裹，包含多个键值对。
> 数组（Array）：由 [] 包裹，包含多个元素。
```json
{
  "name": "Alice",
  "age": 30,
  "is_student": false,
  "courses": ["Math", "Physics", "Computer Science"]
}

```
- 写入json文件
```python
import json

# 假设有一个包含多个员工信息的字典列表
employees = [
    {"name": "Alice", "age": 30, "department": "研发部"},
    {"name": "Bob", "age": 25, "department": "市场部"},
    {"name": "Charlie", "age": 35, "department": "人事部"}
]

# 打开 JSON 文件进行写入
with open('employees.json', 'w', encoding='utf-8') as f:
    # 使用 for 循环将每个员工的数据写入 JSON 文件
    for employee in employees:
        # 写入数据，每次循环写入一个员工信息
        json.dump(employee, f, ensure_ascii=False, indent=4)
        f.write("\n")  # 每个员工信息写完后换行，确保每个 JSON 对象占一行


```

#### pytorch 文件
1. .pt 文件
- 用途：.pt 文件是 PyTorch 模型的通用文件格式，通常用于存储训练好的 模型（包括网络结构和权重）。
- 存储整个模型，包括模型的结构和训练的权重。

- 适用于PyTorch 的 模型保存与加载。

2. .pth 文件
- 用途：.pth 文件也是用于存储 PyTorch 模型的 权重（weights），但是一般只保存模型的 权重参数。有时候它也保存优化器的状态（比如 Adam 优化器的历史梯度、动量等）。

- **存储**模型权重，但不包括模型的结构。

- **存储**模型的权重和优化器状态，通常用于训练中断后的继续训练。

3. 使用方法
- 保存和加载整个模型
```python
import torch

# 假设你有一个已经训练好的模型
model = YourModel()

# 保存整个模型（包括结构和权重）
torch.save(model, 'model.pt')

# 加载整个模型（结构 + 权重）
model = torch.load('model.pt')
model.eval()  # 切换到评估模式（推理）
```
- 保存和加载.pth文件
```python
import torch

# 假设你有一个已经训练好的模型
model = YourModel()

# 保存模型权重
torch.save(model.state_dict(), 'model_weights.pth')

# 假设你有相同的模型结构，加载权重
model = YourModel()

# 加载保存的权重
model.load_state_dict(torch.load('model_weights.pth'))

# 切换到评估模式（推理）
model.eval()

```

### git使用方法
用了这么多git，还是不太会用，今天总结一下应该就会好很多啦
1. **基本配置**
```bash
git config --global user.name "Your Name"  # 设置用户名
git config --global user.email "youremail@example.com"  # 设置邮箱
你可以通过 git config --list 查看当前的配置。
```

2. 仓库初始化
```bash
# 初始化当前目录为一个 Git 仓库
git init

# 以main为分支创建仓库
git init -b main

# 克隆远程仓库
git clone https://github.com/username/repository.git
# 或者通过SSH
git clone git@github.com:HeisenbergsPizza/HHM.git

```


3. **git基本操作**
- `git status` : 查看当前状态
- `git add`: 提交到暂存区
```bash
# 添加单个文件
git add filename

# 添加所有文件
git add .
```
- `git commit`:提交更改
```bash
git commit -m "提交信息"
```
- `git log`：查看提交记录

4. **git 分支管理**
- `git branch`：查看或者创建新分支
```bash
# 查看所有分支
git branch
# 创建新分支
git branch new-branch
```
- `git checkout`:切换分支
- `git merge`:合并分支
```bash
# 切换到目标分支（如 master）
git checkout master

# 合并其他分支（如 new-branch）到当前分支
git merge new-branch
```
- `git branch -d`:删除分支
- `git branch -m master main`：改名


5. git 远程操作
- `git remote add`:添加远程仓库
```bash
# 查看远程仓库
git remote -v
# 添加远程仓库
git remote add origin https://github.com/username/repository.git

```
- `git pull`:拉取远程仓库
```bash
git pull origin main

# 拉取不同分支
git pull origin feature-branch
```
- `git push`:推送给远程仓库
```bash
git push origin main
# 推送不同分支
git push -u origin new-branch
```

6. 回退操作
`git rebase`
```bash
# 回退到某个提交，但保留暂存区和工作区的更改。
git reset --soft <commit>：
# 回退到某个提交，保留工作区的更改，但取消暂存区的更改（这是默认模式）。
git reset --mixed <commit>：
# 回退到某个提交，并丢弃工作区和暂存区的所有更改
git reset --hard <commit>
# 回退到上一次提交
git reset --hard HEAD~1
# 撤销暂存区的修改
git reset <file>

```

7. 简单工作流
- 首先，init和链接好远端仓库
- 其次，对于不同的任务创建不同的分支
- 接着，完成了一个target，例如DIP写完了一个函数，就add file，等整个写完了再commit。
- 每次开始新写一些东西，git pull --rebase origin main，保证自己的main分支最新。
- 当`git push origin main`时，先`git pull origin main`,看看**main**分支上有什么改变，接着`git switch new_branch`,`git rebase main`看看自己分支的功能还能正常进行吗？
- 能正常进行之后，再`git push -f origin new_branch`,之后再上github pull and request




## 配环境总结
1. cuda使用技巧
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
1. **虚拟环境**
- 所有模型使用的时候都需要配一个新环境，除了非常新的模型，因为大部分模型的年代都有点久远，现在的libliary多少有些不兼容。
```python
conda create -n fwx python==3.9
```


## 模型总结

## 总结