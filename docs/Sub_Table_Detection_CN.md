# 子表检测算法详解

## 概述

子表检测是电子表格编码器中的关键功能，用于识别一个工作表中包含的多个独立表格。许多真实世界的电子表格在同一个sheet中包含多个独立的数据表，需要分别处理和分析。

**代码位置**: `spreadsheet-normalizer/src/encoder/spreadsheet_encoder.py`

---

## 检测策略概览

编码器使用**三层检测策略**，按优先级顺序尝试：

```
策略1: 标题模式检测（Title-based Detection）
   ↓ 失败
策略2: 颜色分组检测（Color-based Detection）
   ↓ 失败
策略3: 空白行分离检测（Blank-row Separation）
   ↓ 失败
回退: 整个区域作为单一表格
```

---

## 策略1: 标题模式检测

### 函数: `detect_table_by_title_rows()`

**位置**: `spreadsheet_encoder.py:301-354`

### 工作原理

扫描前几列，查找符合表格标题模式的行，然后以这些标题行为分界点切分表格。

### 识别的标题模式

使用正则表达式识别常见的表格标题：

```python
title_pattern = re.compile(
    r'(表|Table|统计表|統計表|KeyStat|Sheet|工作表|Worksheet)\s*[\d\.]+',
    re.IGNORECASE
)
```

**匹配示例**:
- ✅ "表1", "表 1", "表3.1"
- ✅ "Table 1", "Table 3.1"
- ✅ "KeyStat_1", "KeyStat_2"
- ✅ "統計表 1", "统计表1"
- ✅ "Sheet 1", "工作表2"

### 算法步骤

```python
# 步骤1: 扫描标题
title_rows = []
scan_cols = min(3, total_columns)  # 只扫描前3列

for row in range(min_row, max_row + 1):
    for col_offset in range(scan_cols):
        cell_value = sheet.cell(row, col).value

        if title_pattern.search(cell_value):
            title_rows.append(row)
            break  # 找到就跳到下一行

# 步骤2: 将标题行转换为表格段
segments = []
for i, start_row in enumerate(title_rows):
    if i < len(title_rows) - 1:
        end_row = title_rows[i + 1] - 1
    else:
        end_row = max_row

    # 过滤太小的段（至少3行）
    if end_row - start_row >= 2:
        segments.append((start_row, end_row))

# 步骤3: 为每个段确定列边界
for start_row, end_row in segments:
    # 统计每列的非空单元格数
    col_counts = {}
    for row in range(start_row, end_row + 1):
        for col in range(min_col, max_col + 1):
            if cell_is_nonempty(row, col):
                col_counts[col] += 1

    min_col = min(col_counts.keys())
    max_col = max(col_counts.keys())

    regions.append(SheetRegion(start_row, end_row, min_col, max_col))
```

### 示例

**输入表格**:
```
     A           B         C         D
1   表1 销售数据
2   产品        数量      价格
3   苹果        100       5.0
4   橘子        200       3.0
5
6   Table 2 库存信息
7   Item        Stock     Location
8   Apple       50        A1
9   Orange      150       A2
```

**检测结果**:
```
找到2个标题行: [1, 6]

表格段:
1. 行 1-5  （表1 销售数据）
2. 行 6-9  （Table 2 库存信息）
```

---

## 策略2: 颜色分组检测

### 函数: `detect_color_based_sub_tables()`

**位置**: `spreadsheet_encoder.py:1357-1416`

### 工作原理

通过分析表头行的颜色变化来识别不同的子表。假设不同子表的标题行使用不同的颜色。

### 检测逻辑

```python
def detect_color_based_sub_tables(sheet, header_row=1, min_gap=1):
    color_groups = []
    current_group = None
    empty_count = 0

    for col in range(1, max_column + 1):
        cell = sheet.cell(row=header_row, column=col)
        color = get_fill_color(cell)

        # 判断是否为"空"列
        is_empty = (
            cell.value is None OR
            (cell.value is blank string) OR
            (color is white/black/none) OR
            (entire column is empty)
        )

        if is_empty:
            empty_count += 1
            if empty_count >= min_gap:
                # 遇到足够的空列，结束当前组
                if current_group:
                    save_group(current_group)
                    current_group = None
        else:
            empty_count = 0

            if current_group is None:
                # 开始新组
                current_group = {
                    'start_col': col,
                    'end_col': col,
                    'colors': {color}
                }
            else:
                # 检查颜色是否改变
                if color != current_group['colors']:
                    if len(current_group['colors']) > 0:
                        # 颜色变化，可能是新表格
                        if current_group['end_col'] - current_group['start_col'] >= 2:
                            save_group(current_group)
                            current_group = new_group(col, color)
                        else:
                            # 组太小，继续扩展
                            extend_group(current_group, col, color)
                else:
                    extend_group(current_group, col, color)
```

### 颜色判断

```python
def get_fill_color(cell):
    # 提取填充颜色
    if cell.fill and cell.fill.patternType == 'solid':
        if cell.fill.fgColor and cell.fill.fgColor.rgb:
            return str(cell.fill.fgColor.rgb)
    return 'none'

# 排除的颜色（视为无颜色）:
# - 'FFFFFF' (白色)
# - '000000' (黑色)
# - 'none' (无填充)
```

### 示例

**输入表格**:
```
     A          B          C          D          E          F
     ┌──────────蓝色背景────────┐    ┌──────────绿色背景────────┐
1    │  产品    │  数量  │       │    │  仓库   │  库存  │
     └──────────────────────────┘    └──────────────────────────┘
2    苹果       100               橙子       200
3    橘子       150               苹果       180
```

**检测结果**:
```
颜色组1: 列A-C (蓝色背景)
颜色组2: 列E-F (绿色背景)

子表1: 列A-C, 行1-3
子表2: 列E-F, 行1-3
```

---

## 策略3: 空白行分离检测

### 函数: `detect_table_regions()` - 空白行逻辑

**位置**: `spreadsheet_encoder.py:357-474`

### 工作原理

如果前两种策略都失败，使用传统的空白行/列分离方法。

### 算法步骤

```python
# 步骤1: 统计每行每列的非空单元格数
row_has = {}  # row_id -> count
col_has = {}  # col_id -> count

for row, col, value in iterate_all_cells(region):
    if value is not None and value.strip() != "":
        row_has[row] += 1
        col_has[col] += 1

# 步骤2: 找到连续的非空行段
nonempty_rows = sorted(row_has.keys())  # [1, 2, 3, 7, 8, 9, 10]

row_segments = []
start = prev = nonempty_rows[0]

for row in nonempty_rows[1:]:
    if row == prev + 1:
        # 连续
        prev = row
    else:
        # 不连续，保存前一段
        row_segments.append((start, prev))
        start = prev = row

row_segments.append((start, prev))
# 结果: [(1, 3), (7, 10)]  表示两个行段

# 步骤3: 同样方法找列段
col_segments = find_contiguous_segments(col_has)

# 步骤4: 行段×列段 = 候选区域
regions = []
for (r_start, r_end) in row_segments:
    for (c_start, c_end) in col_segments:
        region = SheetRegion(r_start, r_end, c_start, c_end)

        # 过滤：至少要有min_nonempty_cells个非空单元格
        if count_nonempty(region) >= min_nonempty_cells:
            regions.append(region)
```

### 示例

**输入表格**:
```
     A       B       C       D       E
1   产品    数量    价格
2   苹果    100     5.0
3   橘子    200     3.0
4                                    ← 空行
5                                    ← 空行
6   仓库    库存    位置
7   A1      50      北京
8   A2      150     上海
```

**检测过程**:
```
非空行: [1, 2, 3, 6, 7, 8]
非空列: [1, 2, 3]

行段识别:
- 段1: 行1-3 (连续)
- 段2: 行6-8 (连续)

列段识别:
- 段1: 列1-3 (连续)

交叉组合:
- 区域1: 行1-3, 列1-3
- 区域2: 行6-8, 列1-3

最终: 检测到2个子表
```

---

## 连通组件辅助检测

### 函数: `extract_sub_tables_from_components()`

**位置**: `spreadsheet_encoder.py:1270-1354`

### 工作原理

利用前面提到的**连通组件算法**来检测子表。如果连通组件算法检测到多个独立的组件，每个组件可能代表一个子表。

### 算法步骤

```python
def extract_sub_tables_from_components(components, sheet, min_gap=2, min_size=4):
    # 步骤1: 过滤太小的组件
    components = [c for c in components if len(c) >= min_size]

    # 步骤2: 为每个组件计算边界框和特征
    component_info = []
    for component in components:
        # 计算边界
        min_row = min(all rows in component)
        max_row = max(all rows in component)
        min_col = min(all cols in component)
        max_col = max(all cols in component)

        # 提取标题区域的颜色特征
        header_colors = set()
        for col in range(min_col, min(max_col + 1, min_col + 10)):
            for row in range(min_row, min(max_row + 1, min_row + 3)):
                color = get_fill_color(sheet.cell(row, col))
                if color and color != 'FFFFFF':
                    header_colors.add(color)

        component_info.append({
            'bounds': (min_row, max_row, min_col, max_col),
            'colors': header_colors,
            'size': len(component)
        })

    # 步骤3: 按列位置排序（从左到右）
    component_info.sort(key=lambda x: (x['min_col'], x['min_row']))

    # 步骤4: 合并重叠/邻近的组件
    sub_tables = []
    for info in component_info:
        merged = False

        for existing_table in sub_tables:
            # 检查列重叠
            col_overlap = not (
                info['max_col'] < existing_table.min_col - min_gap OR
                info['min_col'] > existing_table.max_col + min_gap
            )

            # 检查行重叠
            row_overlap = not (
                info['max_row'] < existing_table.min_row - min_gap OR
                info['min_row'] > existing_table.max_row + min_gap
            )

            if col_overlap and row_overlap:
                # 合并到现有表格
                existing_table.expand_to_include(info)
                merged = True
                break

        if not merged:
            # 创建新子表
            sub_tables.append(SubTableRegion(info))

    return sub_tables
```

### 合并逻辑图示

```
组件1:  ┌────┐
        │ A  │  行1-3, 列1-2
        └────┘

组件2:      ┌────┐
            │ B  │  行2-4, 列3-4
            └────┘

min_gap = 1

判断: 列2和列3相邻（gap=1 <= min_gap）
      行2-3重叠

结果: 合并 → ┌──────┐
            │  AB  │  行1-4, 列1-4
            └──────┘
```

---

## 主检测方法

### 函数: `_detect_sub_tables()`

**位置**: `spreadsheet_encoder.py:1977-2035` (在SpreadsheetEncoder类中)

### 完整流程

```python
def _detect_sub_tables(self, sheet):
    logger.info("🔍 开始检测子表...")

    sub_tables = []

    # ========== 方法1: 颜色分组检测 ==========
    if self.detect_color_subtables:
        all_color_groups = []

        # 尝试多个可能的标题行（前5行）
        for header_row in range(1, min(6, sheet.max_row + 1)):
            groups = detect_color_based_sub_tables(
                sheet, header_row, self.subtable_min_gap
            )

            if groups:
                all_color_groups.extend(groups)

        if all_color_groups:
            # 去重并合并重复的组
            unique_groups = merge_duplicate_groups(all_color_groups)

            # 为每个颜色组扫描数据范围
            for group in unique_groups:
                min_row, max_row = scan_data_range(
                    sheet, group['start_col'], group['end_col']
                )

                sub_tables.append(SubTableRegion(
                    min_row=min_row,
                    max_row=max_row,
                    min_col=group['start_col'],
                    max_col=group['end_col'],
                    header_colors=group['colors']
                ))

            # 清理包含关系
            sub_tables = self._remove_contained_tables(sub_tables)

    # ========== 方法2: 连通组件检测 ==========
    if not sub_tables:
        graph = build_connectivity_graph(sheet, None)

        if graph and len(graph) >= 4:
            components = find_connected_components(graph, min_component_size=4)

            if len(components) > 1:
                sub_tables = extract_sub_tables_from_components(
                    components, sheet, self.subtable_min_gap
                )

    return sub_tables
```

---

## 去重和清理

### 函数: `_remove_contained_tables()`

**位置**: `spreadsheet_encoder.py:2037-2094`

### 目的

移除被其他表格包含的表格，以及处理错误的重叠检测。

### 清理规则

```python
def _remove_contained_tables(sub_tables):
    filtered = []

    for table_a in sub_tables:
        should_keep = True

        # 规则1: 移除被完全包含的表格
        for table_b in sub_tables:
            if table_a == table_b:
                continue

            # A的列范围完全在B内，且A更小
            if (table_a.min_col >= table_b.min_col and
                table_a.max_col <= table_b.max_col and
                table_a.cell_count < table_b.cell_count):
                should_keep = False
                break

        if not should_keep:
            continue

        # 规则2: 移除与多个不相交表格重叠的表格
        overlapping_tables = find_overlapping_tables(table_a, sub_tables)

        if len(overlapping_tables) >= 2:
            # 检查这些重叠表格之间是否不相交
            distinct_overlaps = count_distinct_pairs(overlapping_tables)

            if distinct_overlaps > 0:
                # table_a可能是误检，跨越了多个独立表格
                should_keep = False

        if should_keep:
            filtered.append(table_a)

    return filtered
```

### 示例

**场景1: 包含关系**
```
表A: 行1-10, 列1-5, 50个单元格
表B: 行2-8,  列2-4, 21个单元格  ← B完全在A内

结果: 保留A，移除B
```

**场景2: 跨表重叠（误检）**
```
表A: 行1-5, 列1-3
表B: 行1-5, 列5-7
表C: 行1-5, 列2-6  ← C同时与A和B重叠，但A和B不重叠

结果: 保留A和B，移除C（C可能是误检）
```

---

## 配置参数

### 在 `config.yaml` 中配置

```yaml
encoder:
  # 基础锚点邻域
  anchor_neighborhood: 2

  # 是否启用子表检测
  detect_subtables: true

  # 是否启用颜色分组子表检测
  detect_color_subtables: true

  # 子表之间的最小间隔（列数）
  subtable_min_gap: 2

  # 区域最小非空单元格数
  min_nonempty_cells_for_region: 8
```

### 在 SpreadsheetEncoder 中使用

```python
encoder = SpreadsheetEncoder({
    'detect_subtables': True,
    'detect_color_subtables': True,
    'subtable_min_gap': 2,
    'min_nonempty_cells_for_region': 8
})
```

---

## 完整检测流程图

```
开始
  ↓
加载Excel文件
  ↓
对于每个Sheet:
  ↓
┌─────────────────────────────────┐
│ 策略1: 标题模式检测              │
│ detect_table_by_title_rows()    │
└─────────────────────────────────┘
  ↓
是否找到 >1 个表格？
  ├─ 是 → 返回检测结果
  │
  └─ 否 ↓
┌─────────────────────────────────┐
│ 策略2: 空白行分离                │
│ detect_table_regions()          │
│ (空白行/列分割逻辑)              │
└─────────────────────────────────┘
  ↓
是否找到 >1 个区域？
  ├─ 是 → 返回区域列表
  │
  └─ 否 ↓
┌─────────────────────────────────┐
│ 方法3a: 颜色分组检测             │
│ detect_color_based_sub_tables() │
└─────────────────────────────────┘
  ↓
是否找到颜色组？
  ├─ 是 → 返回子表
  │
  └─ 否 ↓
┌─────────────────────────────────┐
│ 方法3b: 连通组件检测             │
│ extract_sub_tables_from_        │
│ components()                    │
└─────────────────────────────────┘
  ↓
是否找到 >1 个组件？
  ├─ 是 → 返回子表
  │
  └─ 否 ↓
返回整个工作表作为单一表格
```

---

## 实际应用示例

### 示例1: 标题模式检测成功

**输入**:
```excel
1  表1 销售数据
2  产品    数量
3  苹果    100
4
5  表2 库存信息
6  产品    库存
7  苹果    50
```

**输出**:
```python
检测方法: 标题模式
找到2个子表:
  - 子表1: 行1-3 (表1 销售数据)
  - 子表2: 行5-7 (表2 库存信息)
```

---

### 示例2: 颜色分组检测成功

**输入**:
```excel
     A(蓝)  B(蓝)    C     D(绿)  E(绿)
1    产品   数量          仓库   库存
2    苹果   100           A1     50
```

**输出**:
```python
检测方法: 颜色分组
找到2个子表:
  - 子表1: 列A-B (蓝色标题)
  - 子表2: 列D-E (绿色标题)
```

---

### 示例3: 空白行分离

**输入**:
```excel
1  产品  数量
2  苹果  100
3
4
5  仓库  库存
6  A1    50
```

**输出**:
```python
检测方法: 空白行分离
找到2个区域:
  - 区域1: 行1-2
  - 区域2: 行5-6
```

---

## 总结

子表检测算法使用**多策略层级检测**方法：

1. **标题模式优先** - 最可靠，基于明确的文本标记
2. **颜色分组次之** - 基于视觉样式，适合设计规范的表格
3. **空白分离保底** - 传统方法，通用但可能不够精确
4. **连通组件辅助** - 利用图算法，处理复杂布局

这种多层策略确保了在各种复杂场景下都能有效识别子表，提高了电子表格处理的鲁棒性。

**核心优势**:
- ✅ 支持中英文标题
- ✅ 支持颜色分组
- ✅ 支持复杂布局
- ✅ 自动去重和清理
- ✅ 可配置的参数
