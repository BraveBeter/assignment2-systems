## 代码编写原则 (Code Principles)

### 1. 极致精简 (Conciseness)
* **追求一行解决**：在确保功能正确与无效率损耗的前提下，优先使用简洁且表达力强的语法（如推导式、链式调用等）。能用一行代码表达的逻辑绝不拆成多行。
* **零冗余代码**：严禁出现未使用的变量、重复的判断逻辑或死代码（Dead Code）。

### 2. 模块化与高复用 (Modularity & Reusability)
* **抽离公共逻辑**：将所有与当前核心业务逻辑无关、或具备普适性的通用功能（如：数据加载/保存、格式转换、通用性能工具、日志记录等）强行抽离为独立的工具模块/函数。可以写入`student_scripts/a2k/utils.py`  
* **单脚本精简**：保持单个 Python 脚本专注且轻量，严禁编写过长、功能糅杂的“万能脚本”（Monolithic Script）。
> 在实现函数之前，先查看`student_scripts/a2k/utils.py`是否已经实现，有则直接复用。

### 3. 高效与优雅 (Performance & Elegance)
* **算法与原生优先**：除非是任务指定要求，否则优先选用高效的算法、数据结构及语言原生/高性能库（如 `itertools`、`numpy`、`torch` 内置优化算子。
* **自解释与优雅**：代码结构清晰、命名精准，用代码本身说明意图，减少不必要的陈述性注释。


Markdown
### 💡 代码风格示范

❌ **拒绝（冗长/无复用）：**
```python
def process_data(data_list):
    res = []
    for item in data_list:
        if item > 0:
            res.append(item * 2)
    return res
    
```
✅ 推荐（简洁/表达力强）：
```Python
def process_data(data_list: list[int]) -> list[int]:
    return [x * 2 for x in data_list if x > 0]
```