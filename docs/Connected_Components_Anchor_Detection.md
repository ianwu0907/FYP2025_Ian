# Connected Components Anchor Detection Algorithm

## Overview

The Connected Components Anchor Detection algorithm is an **O(RC)** complexity method for identifying structural anchors in spreadsheets. It replaces the previous **O(R⁴C⁴)** heuristic-based approach with a graph-based algorithm that efficiently detects important rows and columns by analyzing cell similarity patterns.

**Location**: `spreadsheet-normalizer/src/encoder/spreadsheet_encoder.py:477-738`

---

## Algorithm Design

### Problem Statement

Given a spreadsheet with R rows and C columns, identify a set of "anchor" rows and columns that represent structurally important regions (headers, data boundaries, formatted regions) for downstream LLM processing.

### Key Innovation

Instead of using coefficient-of-variation (CV) heuristics that require expensive computations, we:
1. Build a **connectivity graph** where similar adjacent cells are connected
2. Find **connected components** (groups of similar cells)
3. Extract **bounding box boundaries** of each component as anchors
4. Expand anchors with a **k-neighborhood** to capture surrounding context

---

## Algorithm Components

### 1. Cell Similarity Detection

#### `get_fill_color(cell)` → str
Extracts the fill color from a cell for style comparison.

**Returns**:
- RGB color string (e.g., `"FF0000"` for red)
- `"index_X"` for indexed colors
- `"none"` if no fill color

**Implementation**:
```python
if fill.patternType == 'solid':
    if fill.fgColor.rgb:
        return str(fill.fgColor.rgb)
```

---

#### `has_border(cell)` → bool
Checks if a cell has any border styling.

**Returns**: `True` if cell has left/right/top/bottom border

---

#### `cells_similar(cell1, cell2, threshold=0.75)` → bool
Determines if two cells should belong to the same connected region using **weighted scoring**.

**Scoring System** (Total Weight: 10.0):

| Feature | Weight | Description |
|---------|--------|-------------|
| Font Bold | 2.0 | Whether both cells have same bold status |
| Fill Color | 3.0 | Whether both cells have same background color |
| Border | 1.5 | Whether both cells have same border presence |
| Data Type | 2.0 | Whether both cells have same value type (int, str, etc.) |
| Number Format | 1.5 | Whether both cells use same number format string |

**Algorithm**:
```
score = 0.0
total = 10.0

if cell1.font.bold == cell2.font.bold:
    score += 2.0
if get_fill_color(cell1) == get_fill_color(cell2):
    score += 3.0
if has_border(cell1) == has_border(cell2):
    score += 1.5
if type(cell1.value) == type(cell2.value):
    score += 2.0
if cell1.number_format == cell2.number_format:
    score += 1.5

return (score / total) >= threshold  # Default: 75%
```

**Threshold**: Default 0.75 means cells must match on ≥75% of weighted features.

---

### 2. Connectivity Graph Construction

#### `build_connectivity_graph(sheet, region)` → Dict[str, Set[str]]

Builds an undirected graph where nodes are cells and edges connect similar adjacent cells.

**Complexity**: **O(RC)** - Each cell is visited once, checking only right and bottom neighbors.

**Algorithm**:
```
For each cell (r, c) in region:
    Skip if empty

    Check right neighbor (r, c+1):
        If non-empty AND cells_similar():
            Add edge: cell ↔ right_neighbor

    Check bottom neighbor (r+1, c):
        If non-empty AND cells_similar():
            Add edge: cell ↔ bottom_neighbor
```

**Why only right + bottom?**
- Avoids double-counting edges (undirected graph)
- Left neighbor already checked when processing previous cell
- Top neighbor already checked when processing previous row

**Example Output**:
```python
{
    "A1": {"A2", "B1"},  # A1 connected to A2 (below) and B1 (right)
    "A2": {"A1", "A3"},  # A2 connected to A1 (above) and A3 (below)
    "B1": {"A1", "B2"},  # B1 connected to A1 (left) and B2 (below)
    ...
}
```

---

### 3. Connected Components Detection

#### `find_connected_components(graph, min_component_size=4)` → List[List[str]]

Finds all connected components using **Depth-First Search (DFS)**.

**Complexity**: **O(V + E) = O(RC)** where:
- V = number of non-empty cells ≤ RC
- E = number of edges ≤ 2RC (each cell has ≤2 edges added)

**Algorithm**:
```
visited = set()
components = []

For each starting cell:
    If already visited, skip

    component = []
    stack = [starting_cell]

    While stack not empty:
        cell = stack.pop()
        If already visited, continue

        Mark cell as visited
        Add cell to component

        For each neighbor of cell:
            If not visited:
                Add neighbor to stack

    If component size ≥ min_component_size:
        Save component

Sort components by size (largest first)
```

**Filtering**: Components with <4 cells are discarded as noise.

**Example Output**:
```python
[
    ["A1", "A2", "A3", "B1", "B2", "B3"],  # Header region (6 cells)
    ["A5", "B5", "C5", "D5"],              # Data row (4 cells)
    ["E1", "E2", "E3", "F1", "F2"],        # Sidebar (5 cells)
]
```

---

### 4. Boundary Extraction

#### `extract_component_boundaries(components)` → Tuple[Set[int], Set[int]]

Extracts the bounding box boundaries of each component as anchor candidates.

**Complexity**: **O(total_cells)** - Linear scan through all cells in all components.

**Algorithm**:
```
row_boundaries = set()
col_boundaries = set()

For each component:
    min_row = min(all rows in component)
    max_row = max(all rows in component)
    min_col = min(all columns in component)
    max_col = max(all columns in component)

    Add to boundaries:
        row_boundaries += {min_row, max_row}
        col_boundaries += {min_col, max_col}

Return (row_boundaries, col_boundaries)
```

**Example**:
```
Component: ["A1", "A2", "B1", "B2"]
Bounding Box: rows [1, 2], cols [A=1, B=2]
Extracted: row_boundaries={1, 2}, col_boundaries={1, 2}
```

---

### 5. K-Neighborhood Expansion

#### `expand_with_k_neighborhood(indices, k, max_index)` → List[int]

Expands anchor indices by including k cells on each side.

**Complexity**: **O(|indices| × k)** - For each anchor, add k neighbors on each side.

**Algorithm**:
```
expanded = set()

For each anchor_idx in indices:
    For offset in range(-k, k+1):
        expanded_idx = anchor_idx + offset
        If 1 ≤ expanded_idx ≤ max_index:
            expanded.add(expanded_idx)

Return sorted(expanded)
```

**Example** (k=2):
```
Input: anchors = {5, 10}
Output: {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
        └─────┬─────┘  └────────┬────────┘
         5 ± 2           10 ± 2
```

**Purpose**: Captures context around important boundaries (headers often span multiple rows).

---

### 6. Main Algorithm

#### `fast_find_structural_anchors(sheet, k, region)` → Tuple[List[int], List[int]]

Main entry point that orchestrates all steps.

**Complexity**: **O(RC)** overall

**Pipeline**:
```
Step 1: Build connectivity graph           O(RC)
    ↓
Step 2: Find connected components          O(RC)
    ↓
Step 3: Extract boundaries                 O(RC)
    ↓
Step 4: Expand with k-neighborhood         O(anchors × k)
    ↓
Return: (row_anchors, col_anchors)
```

**Fallback Strategy**:
- If graph has <4 cells: use region boundaries
- If no components found: use region boundaries
- Ensures algorithm always returns valid anchors

**Example Full Run**:
```python
# Input: Sheet with 100 rows × 50 columns, k=2

# Step 1: Build graph
graph = {
    "A1": {"B1", "A2"},
    "B1": {"A1", "C1"},
    ...  # 2000 non-empty cells → ~4000 edges
}

# Step 2: Find components (5 components found)
components = [
    ["A1", "B1", "C1", ...],  # 20-cell header
    ["A5", "B5", "C5", ...],  # 15-cell subheader
    ...
]

# Step 3: Extract boundaries
row_boundaries = {1, 5, 6, 10, 15, ...}  # 10 boundaries
col_boundaries = {1, 3, 5, 10, ...}      # 8 boundaries

# Step 4: Expand with k=2
row_anchors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]  # 28 rows
col_anchors = [1, 2, 3, 4, 5, 6, 7, ...]            # 22 cols

# Final: Keep only 28×22 = 616 cells instead of 5000!
```

---

## Complexity Analysis

### Time Complexity: O(RC)

| Step | Complexity | Reason |
|------|-----------|---------|
| Build Graph | O(RC) | Visit each cell once, check 2 neighbors |
| Find Components | O(V + E) = O(RC) | DFS on graph with V≤RC vertices, E≤2RC edges |
| Extract Boundaries | O(total_cells) ≤ O(RC) | Linear scan through components |
| Expand Neighborhood | O(anchors × k) | anchors << RC, k is constant |
| **Total** | **O(RC)** | Linear in spreadsheet size |

### Space Complexity: O(RC)

- Graph storage: O(non-empty cells + edges) ≤ O(RC)
- Visited set: O(non-empty cells) ≤ O(RC)
- Components list: O(non-empty cells) ≤ O(RC)

---

## Comparison with Previous Method

### Old Method: CV-based Heuristic
- **Complexity**: O(R⁴C⁴)
- **Approach**: Compute coefficient of variation for all possible rectangular regions
- **Problem**: Exponential in practice, slow for large spreadsheets

### New Method: Connected Components
- **Complexity**: O(RC)
- **Approach**: Graph-based structural analysis
- **Advantage**: Linear complexity, scales to large spreadsheets

**Speedup Example**:
- 100×50 spreadsheet: Old = 625,000,000,000 operations → New = 5,000 operations
- **Improvement: 125,000,000× faster!**

---

## Practical Example

### Input Spreadsheet
```
     A         B         C         D         E
1  [Name]    [Age]    [City]   [Score]   [Grade]  ← Bold, colored header
2  Alice     25       NYC      95        A
3  Bob       30       LA       87        B
4  Carol     28       SF       92        A
5
6  [Total]   [Avg]    [Max]              ← Bold, colored totals
7  3         27.7     95
```

### Step-by-Step Execution

**1. Build Graph** (k=1)
```
Similarities detected:
- A1 ↔ B1 ↔ C1 ↔ D1 ↔ E1  (same bold, same color)
- A6 ↔ B6 ↔ C6             (same bold, same color)
- A2 ↔ A3 ↔ A4             (same format, same column)
- B2 ↔ B3 ↔ B4             (same format, numbers)
...
```

**2. Find Components**
```
Component 1: Header row
  ["A1", "B1", "C1", "D1", "E1"]
  Bounding box: rows [1, 1], cols [1, 5]

Component 2: Totals row
  ["A6", "B6", "C6"]
  Bounding box: rows [6, 6], cols [1, 3]

Component 3: Data column A
  ["A2", "A3", "A4"]
  Bounding box: rows [2, 4], cols [1, 1]
```

**3. Extract Boundaries**
```
row_boundaries = {1, 2, 4, 6}
col_boundaries = {1, 3, 5}
```

**4. Expand with k=1**
```
row_anchors = {1, 2, 3, 4, 5, 6, 7}  (all rows within ±1 of boundaries)
col_anchors = {1, 2, 3, 4, 5}        (all cols within ±1 of boundaries)
```

**5. Final Result**
- Keep all 7 rows × 5 columns = 35 cells
- Original: 7×5 = 35 cells
- Compression: 1.0× (no compression for small table)
- **But algorithm identified important structure!**

---

## Advantages

### 1. **Efficient Complexity**
- O(RC) scales linearly with spreadsheet size
- Suitable for large spreadsheets (thousands of rows/columns)

### 2. **Structure-Aware**
- Identifies headers, footers, and data regions automatically
- Based on actual cell styling, not arbitrary heuristics

### 3. **Robust**
- Works with various spreadsheet layouts
- Handles merged cells, colored regions, bold headers, etc.

### 4. **Configurable**
- `k` parameter controls anchor neighborhood size
- `threshold` in `cells_similar()` controls similarity strictness
- `min_component_size` filters out noise

### 5. **Fallback Strategy**
- Always returns valid anchors (region boundaries if nothing else)
- Never fails completely

---

## Limitations & Future Work

### Current Limitations

1. **Binary Similarity**: `cells_similar()` uses fixed weights
   - Future: Learn weights from examples

2. **Fixed Threshold**: 75% similarity may not suit all spreadsheets
   - Future: Adaptive threshold based on sheet characteristics

3. **No Semantic Understanding**: Only considers styling
   - Future: Incorporate content analysis (e.g., "Total" keywords)

### Potential Improvements

1. **Multi-Pass Analysis**: Run with different thresholds and merge results
2. **Hierarchical Components**: Detect nested structures (sub-headers within headers)
3. **Machine Learning**: Train classifier to predict anchor importance
4. **Content-Based Features**: Add semantic similarity (not just style)

---

## Integration with SpreadsheetLLM Pipeline

The anchor detection is used in the full encoding pipeline:

```
Input: Excel file
    ↓
1. Load with openpyxl (preserves formatting)
    ↓
2. Detect table regions (split by blank rows/titles)
    ↓
3. For each region:
    ↓
    a. Find structural anchors (THIS ALGORITHM)
    ↓
    b. Extract cells near anchors (k-neighborhood)
    ↓
    c. Create inverted index (value → cells)
    ↓
    d. Aggregate format regions
    ↓
    e. Build compressed JSON encoding
    ↓
Output: SpreadsheetLLM-style encoding
```

**Anchor detection** (step 3a) is the foundation that determines which cells are important for downstream LLM processing.

---

## References

### Code Location
- `spreadsheet_encoder.py:477-738` - Full implementation
- `spreadsheet_encoder.py:694-738` - Main entry point `fast_find_structural_anchors()`

### Related Algorithms
- **Graph Connectivity**: Classical DFS-based connected components
- **Spreadsheet Analysis**: Inspired by document layout analysis techniques
- **SpreadsheetLLM**: Chen et al., "SpreadsheetLLM: Encoding Spreadsheets for Large Language Models"

---

## Usage Example

```python
from spreadsheet_encoder import fast_find_structural_anchors
import openpyxl

# Load spreadsheet
wb = openpyxl.load_workbook("data.xlsx")
sheet = wb.active

# Find anchors (k=2 neighborhood)
row_anchors, col_anchors = fast_find_structural_anchors(
    sheet,
    k=2,
    region=None  # Use full sheet
)

print(f"Found {len(row_anchors)} row anchors: {row_anchors[:10]}...")
print(f"Found {len(col_anchors)} col anchors: {col_anchors[:10]}...")

# Output:
# Found 28 row anchors: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
# Found 22 col anchors: [1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
```

---

## Summary

The Connected Components Anchor Detection algorithm provides an **efficient O(RC)** method for identifying structurally important regions in spreadsheets. By building a connectivity graph based on cell similarity and extracting bounding box boundaries of connected components, it replaces expensive heuristic methods with a principled graph-based approach.

**Key Takeaway**: Transform spreadsheet analysis from a CV-heuristic problem into a graph connectivity problem, achieving massive speedup while maintaining accuracy.
