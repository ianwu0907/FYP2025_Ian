"""
Spreadsheet Encoder Module - Using SpreadsheetLLM methodology
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from pathlib import Path
import tempfile
import os
import openpyxl
from openpyxl.utils import get_column_letter, column_index_from_string
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")

# ===== SpreadsheetLLM Helper Functions =====

def infer_cell_data_type(cell) -> str:
    if cell.value is None:
        return "empty"
    if isinstance(cell.value, str) and EMAIL_REGEX.match(cell.value):
        return "email"
    data_type = getattr(cell, "data_type", None)
    if data_type == 's':
        return "text"
    elif data_type == 'n':
        return "numeric"
    elif data_type == 'b':
        return "boolean"
    elif data_type == 'd':
        return "datetime"
    elif data_type == 'e':
        return "error"
    elif data_type == 'f':
        if cell.value is not None:
            if isinstance(cell.value, str):
                return "text"
            elif isinstance(cell.value, (int, float)):
                return "numeric"
            elif isinstance(cell.value, bool):
                return "boolean"
            else:
                return "formula_cached_value"
        return "formula"
    elif data_type == 'g':
        if cell.value is not None:
            return "text"
        else:
            return "empty"
    else:
        return "unknown"

def categorize_number_format(number_format_string: str, cell) -> str:
    cell_data_type = infer_cell_data_type(cell)
    if cell_data_type not in ["numeric", "datetime"]:
        return "not_applicable"
    if number_format_string is None or str(number_format_string).lower() == "general":
        if cell_data_type == "datetime":
            return "datetime_general"
        return "general"
    if number_format_string == "@" or str(number_format_string).lower() == "text":
        return "text_format"
    if any(c in str(number_format_string) for c in ['$', '€', '£', '¥']):
        return "currency"
    if '%' in str(number_format_string):
        return "percentage"
    nf_lower = str(number_format_string).lower()
    if 'e+' in nf_lower or 'e-' in nf_lower:
        return "scientific"
    if '#' in nf_lower and '/' in nf_lower and '?' in nf_lower:
        return "fraction"
    date_keywords = ['yyyy', 'yy', 'mmmm', 'mmm', 'mm', 'dddd', 'ddd', 'dd', 'd']
    time_keywords = ['hh', 'h', 'ss', 's', 'am/pm', 'a/p']
    is_date = any(k in nf_lower for k in date_keywords)
    is_time = any(k in nf_lower for k in time_keywords)
    if ':' in str(number_format_string):
        tmp = str(number_format_string).replace('0','').replace('#','').replace(',','').replace('.','')
        if ':' in tmp:
            is_time = True
    if is_date and is_time:
        return "datetime_custom"
    if is_date:
        return "date_custom"
    if is_time:
        return "time_custom"
    if cell_data_type == "numeric":
        if number_format_string in ["0", "#,##0"]:
            return "integer"
        if number_format_string in ["0.00", "#,##0.00", "0.0", "#,##0.0"]:
            return "float"
        return "other_numeric"
    if cell_data_type == "datetime":
        return "other_date"
    return "unknown_format_category"

def get_number_format_string(cell) -> str:
    try:
        nfs = cell.number_format
        if nfs is None or nfs == "":
            return "General"
        return str(nfs)
    except Exception:
        return "General"

def detect_semantic_type(cell) -> str:
    data_type = infer_cell_data_type(cell)
    if data_type == "email":
        return "email"
    nfs = get_number_format_string(cell)
    category = categorize_number_format(nfs, cell)
    nfs_lower = nfs.lower()
    if category == "percentage":
        return "percentage"
    if category == "currency":
        return "currency"
    if category in ["date_custom", "datetime_custom", "datetime_general", "other_date"]:
        if ("yyyy" in nfs_lower or "yy" in nfs_lower) and not any(x in nfs_lower for x in ["m", "d"]):
            return "year"
        return "date"
    if category == "time_custom":
        return "time"
    if category == "scientific":
        return "scientific_notation"
    if data_type == "numeric":
        if isinstance(cell.value, int) or category == "integer":
            return "integer"
        if isinstance(cell.value, float) or category == "float":
            return "float"
        return "numeric"
    return data_type

def split_cell_ref(cell_ref):
    col_str = ''.join(filter(str.isalpha, cell_ref))
    row_str = ''.join(filter(str.isdigit, cell_ref))
    return col_str, int(row_str)

def _merge_refs(refs):
    coords = []
    for ref in sorted(set(refs)):
        try:
            col_letter, row = split_cell_ref(ref)
            col = column_index_from_string(col_letter)
            coords.append((row, col))
        except Exception:
            continue
    cell_set = set(coords)
    processed = set()
    ranges = []
    for row, col in sorted(coords):
        if (row, col) in processed:
            continue
        width = 1
        while (row, col + width) in cell_set and (row, col + width) not in processed:
            width += 1
        height = 1
        expanding = True
        while expanding:
            next_row = row + height
            for w in range(width):
                if (next_row, col + w) not in cell_set or (next_row, col + w) in processed:
                    expanding = False
                    break
            if expanding:
                height += 1
        end_col = col + width - 1
        end_row = row + height - 1
        start_ref = f"{get_column_letter(col)}{row}"
        end_ref = f"{get_column_letter(end_col)}{end_row}"
        if width == 1 and height == 1:
            ranges.append(start_ref)
        else:
            ranges.append(f"{start_ref}:{end_ref}")
        for r in range(row, row + height):
            for c in range(col, col + width):
                processed.add((r, c))
    return ranges

def fast_find_structural_anchors(sheet, k):
    row_values = defaultdict(list)
    col_values = defaultdict(list)
    for r in range(1, min(sheet.max_row+1, 201)):
        for c in range(1, min(sheet.max_column+1, 51)):
            v = sheet.cell(row=r, column=c).value
            if v is not None:
                row_values[r].append(v)
                col_values[c].append(v)

    def coefficient_of_variation(values):
        if not values or len(values) < 2:
            return 0.0
        try:
            mean_val = sum(values) / len(values)
            if mean_val == 0:
                return 0.0
            variance = sum((x - mean_val)**2 for x in values) / len(values)
            std_dev = variance ** 0.5
            return std_dev / abs(mean_val)
        except:
            return 0.0

    row_cv = {}
    for r, vals in row_values.items():
        numeric_vals = []
        for v in vals:
            if isinstance(v, (int, float)):
                numeric_vals.append(v)
        if numeric_vals:
            row_cv[r] = coefficient_of_variation(numeric_vals)

    col_cv = {}
    for c, vals in col_values.items():
        numeric_vals = []
        for v in vals:
            if isinstance(v, (int, float)):
                numeric_vals.append(v)
        if numeric_vals:
            col_cv[c] = coefficient_of_variation(numeric_vals)

    sorted_rows = sorted(row_cv.items(), key=lambda x: x[1], reverse=True)
    sorted_cols = sorted(col_cv.items(), key=lambda x: x[1], reverse=True)

    row_anchors = [r for r, cv in sorted_rows[:k]]
    col_anchors = [c for c, cv in sorted_cols[:k]]

    if not row_anchors:
        row_anchors = list(range(1, min(11, sheet.max_row+1)))
    if not col_anchors:
        col_anchors = list(range(1, min(11, sheet.max_column+1)))

    return sorted(row_anchors), sorted(col_anchors)

def compress_homogeneous_regions(sheet, kept_rows, kept_cols):
    return kept_rows, kept_cols

def aggregate_formats(sheet, format_map):
    aggregated_formats = defaultdict(list)
    processed_cells = set()

    type_nfs_map = defaultdict(list)
    for _, cells in format_map.items():
        for cell_ref in cells:
            try:
                cell = sheet[cell_ref]
            except Exception:
                continue
            nfs = get_number_format_string(cell)
            sem_type = detect_semantic_type(cell)
            key = json.dumps({"type": sem_type, "nfs": nfs}, sort_keys=True)
            type_nfs_map[key].append(cell_ref)

    for key, cells in type_nfs_map.items():
        cells_set = set(cells)
        for start_cell in cells:
            if start_cell in processed_cells:
                continue

            try:
                start_col_letter, start_row = split_cell_ref(start_cell)
                start_col = column_index_from_string(start_col_letter)
            except Exception:
                continue

            best_width = 1
            best_height = 1
            best_area = 1
            best_end_cell = start_cell

            max_width = min(20, sheet.max_column - start_col + 1)
            max_height = min(20, sheet.max_row - start_row + 1)

            for width in range(1, max_width + 1):
                for height in range(1, max_height + 1):
                    valid_rectangle = True
                    for r in range(start_row, start_row + height):
                        for c in range(start_col, start_col + width):
                            cell_ref = f"{get_column_letter(c)}{r}"
                            if cell_ref not in cells_set or cell_ref in processed_cells:
                                valid_rectangle = False
                                break
                        if not valid_rectangle:
                            break

                    if valid_rectangle:
                        area = width * height
                        if area > best_area:
                            best_width = width
                            best_height = height
                            best_area = area
                            best_end_cell = f"{get_column_letter(start_col + width - 1)}{start_row + height - 1}"

            region = start_cell if best_width == 1 and best_height == 1 else f"{start_cell}:{best_end_cell}"
            aggregated_formats[key].append(region)

            for r in range(start_row, start_row + best_height):
                for c in range(start_col, start_col + best_width):
                    processed_cells.add(f"{get_column_letter(c)}{r}")

    return dict(aggregated_formats)

def create_inverted_index(sheet, kept_rows, kept_cols):
    inverted_index = defaultdict(list)
    format_map = defaultdict(list)

    for r in kept_rows:
        for c in kept_cols:
            cell = sheet.cell(row=r, column=c)
            cell_ref = f"{get_column_letter(c)}{r}"

            merged_value = None
            merged_range = None
            for m_range in sheet.merged_cells.ranges:
                if cell.coordinate in m_range:
                    try:
                        merged_value = sheet[m_range.start_cell.coordinate].value
                        merged_range = m_range
                        break
                    except Exception:
                        pass

            try:
                if merged_value is not None:
                    cell_value = str(merged_value) if merged_value is not None else ""
                    inverted_index[cell_value].append(cell_ref)
                elif cell.value is not None:
                    if isinstance(cell.value, (int, float)):
                        cell_value = f"{cell.value}"
                    else:
                        cell_value = str(cell.value)
                    inverted_index[cell_value].append(cell_ref)
            except Exception:
                inverted_index["ERROR_VALUE"].append(cell_ref)

            try:
                format_info = {}
                format_info["font"] = {"bold": getattr(cell.font, 'bold', None)}
                alignment = getattr(cell, 'alignment', None)
                format_info["alignment"] = {"horizontal": getattr(alignment, 'horizontal', None)}
                original_number_format = get_number_format_string(cell)
                inferred_type = infer_cell_data_type(cell)
                category = categorize_number_format(original_number_format, cell)
                format_info["original_number_format"] = original_number_format
                format_info["inferred_data_type"] = inferred_type
                format_info["number_format_category"] = category

                if merged_range is not None:
                    format_info["merged"] = True
                    format_info["merged_range"] = str(merged_range)
                else:
                    format_info["merged"] = False

                format_key = json.dumps(format_info, sort_keys=True)
                format_map[format_key].append(cell_ref)
            except Exception:
                pass

    return dict(inverted_index), dict(format_map)

def create_inverted_index_translation(inverted_index):
    merged_index = {}
    for value, refs in inverted_index.items():
        if value is None or str(value).strip() == "":
            continue
        merged_index[value] = _merge_refs(refs)
    return merged_index

def extract_cells_near_anchors(sheet, row_anchors, col_anchors, k):
    rows_to_keep = set()
    cols_to_keep = set()
    for r in row_anchors:
        for i in range(max(1, r - k), min(sheet.max_row + 1, r + k + 1)):
            rows_to_keep.add(i)
    for c in col_anchors:
        for i in range(max(1, c - k), min(sheet.max_column + 1, c + k + 1)):
            cols_to_keep.add(i)
    return sorted(list(rows_to_keep)), sorted(list(cols_to_keep))

def cluster_numeric_ranges(sheet, format_map):
    numeric_map = {fmt: cells for fmt, cells in format_map.items()
                   if json.loads(fmt).get("inferred_data_type") == "numeric"}

    if not numeric_map:
        for r in range(1, sheet.max_row + 1):
            for c in range(1, sheet.max_column + 1):
                cell = sheet.cell(row=r, column=c)
                if infer_cell_data_type(cell) == "numeric":
                    fmt_key = json.dumps({
                        "type": detect_semantic_type(cell),
                        "nfs": get_number_format_string(cell)
                    }, sort_keys=True)
                    numeric_map.setdefault(fmt_key, []).append(f"{get_column_letter(c)}{r}")

    return aggregate_formats(sheet, numeric_map)

def spreadsheet_llm_encode_with_helpers(excel_path, output_path=None, k=2):
    wb = openpyxl.load_workbook(excel_path, data_only=False)
    sheets_encoding = {}
    compression_metrics = {"sheets": {}}
    overall_orig = overall_anchor = overall_index = overall_format = overall_final = 0

    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        if sheet.max_row <= 1 and sheet.max_column <= 1:
            continue

        # Original tokens
        original_cells = {}
        for r in range(1, sheet.max_row + 1):
            for c in range(1, sheet.max_column + 1):
                v = sheet.cell(row=r, column=c).value
                if v is not None:
                    original_cells[f"{get_column_letter(c)}{r}"] = str(v)
        original_tokens = len(json.dumps(original_cells, ensure_ascii=False))

        # Find anchors
        row_anchors, col_anchors = fast_find_structural_anchors(sheet, k)
        kept_rows, kept_cols = extract_cells_near_anchors(sheet, row_anchors, col_anchors, 0)

        if not kept_rows or not kept_cols:
            kept_rows = list(range(1, min(21, sheet.max_row+1)))
            kept_cols = list(range(1, min(21, sheet.max_column+1)))

        kept_rows, kept_cols = compress_homogeneous_regions(sheet, kept_rows, kept_cols)

        # Anchor tokens
        anchor_cells = {}
        for r in kept_rows:
            for c in kept_cols:
                v = sheet.cell(row=r, column=c).value
                if v is not None:
                    anchor_cells[f"{get_column_letter(c)}{r}"] = str(v)
        anchor_tokens = len(json.dumps(anchor_cells, ensure_ascii=False))

        # Inverted index
        inverted_index, format_map = create_inverted_index(sheet, kept_rows, kept_cols)
        index_tokens = len(json.dumps(inverted_index, ensure_ascii=False))

        # Merge index
        merged_index = create_inverted_index_translation(inverted_index)
        merged_tokens = len(json.dumps(merged_index, ensure_ascii=False))

        # Formats
        aggregated_formats = aggregate_formats(sheet, format_map)
        format_tokens = len(json.dumps(aggregated_formats, ensure_ascii=False))

        # Numeric ranges
        numeric_ranges = cluster_numeric_ranges(sheet, format_map)
        numeric_tokens = len(json.dumps(numeric_ranges, ensure_ascii=False))

        # Final encoding
        sheet_encoding = {
            "structural_anchors": {
                "rows": row_anchors,
                "columns": [get_column_letter(c) for c in col_anchors]
            },
            "cells": merged_index,
            "formats": aggregated_formats,
            "numeric_ranges": numeric_ranges
        }

        final_tokens = len(json.dumps(sheet_encoding, ensure_ascii=False))

        # Metrics
        compression_metrics["sheets"][sheet_name] = {
            "original_tokens": original_tokens,
            "after_anchor_tokens": anchor_tokens,
            "after_inverted_index_tokens": index_tokens,
            "after_merged_index_tokens": merged_tokens,
            "after_format_tokens": format_tokens,
            "numeric_tokens": numeric_tokens,
            "final_tokens": final_tokens,
            "anchor_ratio": (original_tokens / anchor_tokens) if anchor_tokens else 0,
            "index_ratio": (original_tokens / index_tokens) if index_tokens else 0,
            "format_ratio": (original_tokens / format_tokens) if format_tokens else 0,
            "overall_ratio": (original_tokens / final_tokens) if final_tokens else 0
        }

        sheets_encoding[sheet_name] = sheet_encoding
        overall_orig += original_tokens
        overall_anchor += anchor_tokens
        overall_index += index_tokens
        overall_format += format_tokens
        overall_final += final_tokens

    compression_metrics["overall"] = {
        "original_tokens": overall_orig,
        "after_anchor_tokens": overall_anchor,
        "after_inverted_index_tokens": overall_index,
        "after_format_tokens": overall_format,
        "final_tokens": overall_final,
        "overall_ratio": (overall_orig / overall_final) if overall_final else 0
    }

    full_encoding = {
        "file_name": os.path.basename(excel_path),
        "sheets": sheets_encoding,
        "compression_metrics": compression_metrics
    }

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_encoding, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved encoding to {output_path}")

    return full_encoding

def convert_csv_to_xlsx(csv_path, xlsx_path):
    try:
        import pandas as pd
    except:
        raise RuntimeError("pandas required")

    tried = []
    for enc in ['utf-16', 'utf-8', 'latin1']:
        for sep in ['\t', ',', ';', '|']:
            try:
                df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    df.to_excel(xlsx_path, index=False, engine='openpyxl')
                    logger.info(f"Converted CSV -> XLSX (encoding={enc} sep={sep})")
                    return True
            except Exception as e:
                tried.append((enc, sep, str(e)))
                continue

    try:
        df = pd.read_csv(csv_path, encoding='utf-8', sep=None, engine='python')
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        logger.info("Converted CSV -> XLSX using python engine")
        return True
    except Exception as e:
        logger.error("CSV->XLSX conversion failed")
        raise


# ===== Main SpreadsheetEncoder Class =====

class SpreadsheetEncoder:
    """
    SpreadsheetLLM-based encoder for efficient table compression.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k = config.get('anchor_neighborhood', 2)
        logger.info(f"Initialized SpreadsheetEncoder with k={self.k}")

    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load file and convert CSV if needed."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading file: {file_path}")

        # Handle CSV
        if file_path.suffix.lower() == '.csv':
            logger.info("Converting CSV to XLSX...")
            temp_xlsx = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
            temp_xlsx_path = temp_xlsx.name
            temp_xlsx.close()

            try:
                convert_csv_to_xlsx(str(file_path), temp_xlsx_path)
                self._temp_xlsx = temp_xlsx_path
                working_file = temp_xlsx_path
            except Exception as e:
                logger.error(f"CSV conversion failed: {e}")
                # Fallback
                import chardet
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read(10000))
                    encoding = result['encoding']
                for sep in [',', '\t', ';', '|']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:
                            self._working_file = None
                            return df
                    except:
                        continue
                df = pd.read_csv(file_path, encoding=encoding)
                self._working_file = None
                return df
        else:
            working_file = str(file_path)
            self._temp_xlsx = None

        # Load with pandas
        try:
            df = pd.read_excel(working_file, engine='openpyxl')
            logger.info(f"Loaded dataframe with shape {df.shape}")
            self._working_file = working_file
            return df
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def encode(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Encode using SpreadsheetLLM methodology."""
        logger.info("Encoding with SpreadsheetLLM...")

        # Ensure we have a file to work with
        if not hasattr(self, '_working_file') or self._working_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            df.to_excel(temp_file_path, index=False, engine='openpyxl')
            self._working_file = temp_file_path
            self._temp_xlsx = temp_file_path

        # Run encoding
        full_encoding = spreadsheet_llm_encode_with_helpers(
            self._working_file,
            output_path=None,
            k=self.k
        )

        # Extract first sheet
        sheet_name = list(full_encoding['sheets'].keys())[0] if full_encoding['sheets'] else None
        sheet_encoding = full_encoding['sheets'][sheet_name] if sheet_name else {}

        # Create readable text
        encoded_text = self._create_readable_text(sheet_encoding, full_encoding['compression_metrics'])

        # Extract metadata
        metadata = self._extract_metadata(df, sheet_encoding, full_encoding['compression_metrics'])

        # Cleanup temp file
        if self._temp_xlsx and Path(self._temp_xlsx).exists():
            try:
                Path(self._temp_xlsx).unlink()
            except:
                pass

        result = {
            'encoded_text': encoded_text,
            'metadata': metadata,
            'original_shape': df.shape,
            'preview_shape': df.shape,
            'dataframe': df,
            'spreadsheet_llm_encoding': sheet_encoding,
            'compression_metrics': full_encoding['compression_metrics']
        }

        logger.info(f"Encoding complete. Compression: {metadata.get('compression_ratio', 0):.2f}x")
        return result

    def _create_readable_text(self, sheet_encoding: Dict[str, Any],
                              compression_metrics: Dict[str, Any]) -> str:
        """Create LLM-friendly text representation."""
        lines = []

        sheet_metrics = list(compression_metrics.get('sheets', {}).values())[0] if compression_metrics.get('sheets') else {}
        lines.append("="*80)
        lines.append("SPREADSHEET ENCODING (SpreadsheetLLM)")
        lines.append("="*80)
        lines.append(f"Compression Ratio: {sheet_metrics.get('overall_ratio', 0):.2f}x")
        lines.append("")

        # Anchors
        anchors = sheet_encoding.get('structural_anchors', {})
        lines.append("## STRUCTURAL ANCHORS")
        lines.append(f"Key Rows: {anchors.get('rows', [])}")
        lines.append(f"Key Columns: {anchors.get('columns', [])}")
        lines.append("")

        # Cells
        cells = sheet_encoding.get('cells', {})
        lines.append("## CELL VALUES")
        lines.append(f"Unique values: {len(cells)}")
        for idx, (value, ranges) in enumerate(list(cells.items())[:20]):
            ranges_str = ', '.join(ranges[:5])
            if len(ranges) > 5:
                ranges_str += f" (+{len(ranges)-5} more)"
            lines.append(f"  '{value}': {ranges_str}")
        if len(cells) > 20:
            lines.append(f"  ... and {len(cells)-20} more values")
        lines.append("")

        # Formats
        formats = sheet_encoding.get('formats', {})
        lines.append("## FORMAT REGIONS")
        lines.append(f"Format groups: {len(formats)}")
        for fmt_key, ranges in list(formats.items())[:10]:
            try:
                fmt = json.loads(fmt_key)
                ranges_str = ', '.join(ranges[:3])
                if len(ranges) > 3:
                    ranges_str += f" (+{len(ranges)-3})"
                lines.append(f"  Type: {fmt.get('type')}, Format: {fmt.get('nfs')}")
                lines.append(f"    Ranges: {ranges_str}")
            except:
                pass
        lines.append("")

        return "\n".join(lines)

    def _extract_metadata(self, df: pd.DataFrame,
                          sheet_encoding: Dict[str, Any],
                          compression_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata with complete unique values and statistics."""
        sheet_metrics = list(compression_metrics.get('sheets', {}).values())[0] if compression_metrics.get('sheets') else {}

        metadata = {
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
            'compression_ratio': sheet_metrics.get('overall_ratio', 0),
            'compression_stages': {
                'original_tokens': sheet_metrics.get('original_tokens', 0),
                'after_anchors': sheet_metrics.get('after_anchor_tokens', 0),
                'final_tokens': sheet_metrics.get('final_tokens', 0)
            },
            'sample_values': {},
            'columns': {}  # Detailed per-column analysis
        }

        # Configuration
        max_unique_values = 100
        sample_size = min(10000, len(df))
        df_sample = df if len(df) <= sample_size else df.sample(sample_size)

        # Analyze each column in detail
        for col in df.columns:
            col_metadata = self._analyze_column(df[col], df_sample[col], max_unique_values)
            metadata['columns'][col] = col_metadata

            # Keep sample_values for backward compatibility
            non_null = df[col].dropna()
            if len(non_null) > 0:
                metadata['sample_values'][col] = [str(v)[:50] for v in non_null.head(3).tolist()]
            else:
                metadata['sample_values'][col] = []

        # Issues
        issues = []
        for col in df.columns:
            col_str = str(col)
            if any('\u4e00' <= c <= '\u9fff' for c in col_str) and any(c.isalpha() and ord(c) < 128 for c in col_str):
                issues.append(f"Multi-language column: {col}")
        metadata['potential_issues'] = issues

        return metadata

    def _analyze_column(self, full_series: pd.Series, sample_series: pd.Series,
                        max_unique_values: int) -> Dict[str, Any]:
        """Analyze a single column comprehensively to collect all necessary metadata."""
        col_metadata = {
            'dtype': str(full_series.dtype),
            'null_count': int(full_series.isnull().sum()),
            'null_percentage': float(full_series.isnull().sum() / len(full_series) * 100) if len(full_series) > 0 else 0.0,
            'unique_count': int(full_series.nunique()),
            'unique_values': [],
            'value_counts': {},
            'sample_values': [],
            'inferred_type': None,
            'potential_delimiters': [],
            'has_bilingual_content': False,
            'statistics': {}
        }

        # Get unique values (limited to max_unique_values)
        unique_vals = full_series.dropna().unique()
        if len(unique_vals) <= max_unique_values:
            col_metadata['unique_values'] = [str(v) for v in unique_vals]
            col_metadata['unique_values_truncated'] = False

            # Get value counts for low-cardinality columns
            value_counts = full_series.value_counts()
            col_metadata['value_counts'] = {
                str(k): int(v) for k, v in value_counts.head(50).items()
            }
        else:
            col_metadata['unique_values'] = [str(v) for v in unique_vals[:max_unique_values]]
            col_metadata['unique_values_truncated'] = True

        # Sample values
        non_null_sample = sample_series.dropna().head(20)
        col_metadata['sample_values'] = [str(v) for v in non_null_sample]

        # Infer semantic type
        col_metadata['inferred_type'] = self._infer_column_type(full_series, col_metadata)

        # Detect delimiters for potential splitting
        if col_metadata['inferred_type'] == 'string':
            col_metadata['potential_delimiters'] = self._detect_delimiters(
                col_metadata['sample_values']
            )

        # Check for bilingual content
        col_metadata['has_bilingual_content'] = self._check_bilingual(
            col_metadata['sample_values']
        )

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(full_series):
            col_metadata['statistics'] = {
                'min': float(full_series.min()) if not full_series.empty else None,
                'max': float(full_series.max()) if not full_series.empty else None,
                'mean': float(full_series.mean()) if not full_series.empty else None,
                'median': float(full_series.median()) if not full_series.empty else None,
                'std': float(full_series.std()) if not full_series.empty else None
            }

        return col_metadata

    def _infer_column_type(self, series: pd.Series, col_metadata: Dict) -> str:
        """Infer semantic type based on data characteristics.

        Returns: boolean | boolean_with_na | categorical | integer | numeric |
                identifier | date | year | string
        """
        unique_count = col_metadata['unique_count']
        total_count = len(series)
        unique_values = col_metadata.get('unique_values', [])

        # Very low cardinality -> likely categorical or boolean
        if unique_count <= 10:
            unique_vals_lower = [str(v).lower().strip() for v in unique_values]

            # Check for boolean patterns
            boolean_patterns = [
                {'yes', 'no'}, {'y', 'n'}, {'true', 'false'}, {'t', 'f'}, {'1', '0'},
                {'是', '否'}, {'有', '沒有', '无'}, {'有', '没有'}
            ]

            # Check for N/A values
            na_indicators = ['n/a', 'na', 'not applicable', '不適用', '不适用', 'none', 'null']
            has_na_value = any(val in unique_vals_lower for val in na_indicators)

            # Remove N/A from comparison
            actual_vals = set(unique_vals_lower) - set(na_indicators)

            # Check if matches any boolean pattern
            for pattern in boolean_patterns:
                if actual_vals <= pattern and len(actual_vals) >= 2:
                    return 'boolean_with_na' if has_na_value else 'boolean'

            # If only one non-NA value, might still be boolean with missing values
            if len(actual_vals) == 1 and has_na_value:
                single_val = list(actual_vals)[0]
                all_boolean_vals = set()
                for pattern in boolean_patterns:
                    all_boolean_vals.update(pattern)
                if single_val in all_boolean_vals:
                    return 'boolean_with_na'

            # Otherwise categorical
            return 'categorical'

        # High cardinality -> identifier
        if unique_count / total_count > 0.95:
            return 'identifier'

        # Medium cardinality -> categorical
        if unique_count / total_count < 0.5 and unique_count <= 50:
            return 'categorical'

        # Check pandas dtype
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                # Could be year if values are in reasonable range
                if not series.empty:
                    min_val = series.min()
                    max_val = series.max()
                    if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                        return 'year'
                return 'integer'
            return 'numeric'

        # Check for date patterns
        if self._is_date_column(series):
            return 'date'

        return 'string'

    def _detect_delimiters(self, sample_values: List[str]) -> List[Dict[str, Any]]:
        """Detect potential delimiters that could be used for column splitting."""
        if not sample_values or len(sample_values) < 3:
            return []

        delimiters_to_check = [
            (' - ', 'space-dash-space'), ('-', 'dash'),
            (' / ', 'space-slash-space'), ('/', 'slash'),
            (' | ', 'space-pipe-space'), ('|', 'pipe'),
            (':', 'colon'), (';', 'semicolon'), (',', 'comma'),
            ('(', 'open-paren'), (')', 'close-paren')
        ]

        detected = []
        total_values = len(sample_values)

        for delimiter, name in delimiters_to_check:
            count = sum(1 for val in sample_values if delimiter in str(val))
            percentage = (count / total_values * 100) if total_values > 0 else 0

            # Consider it a potential delimiter if it appears in at least 50% of values
            if percentage >= 50:
                split_counts = []
                for val in sample_values:
                    if delimiter in str(val):
                        parts = str(val).split(delimiter)
                        split_counts.append(len(parts))

                # Check if split is consistent
                is_consistent = len(set(split_counts)) == 1 if split_counts else False
                num_parts = split_counts[0] if split_counts and is_consistent else None

                detected.append({
                    'delimiter': delimiter,
                    'name': name,
                    'frequency': count,
                    'percentage': percentage,
                    'is_consistent': is_consistent,
                    'num_parts': num_parts,
                    'sample_split': str(sample_values[0]).split(delimiter) if sample_values and delimiter in str(sample_values[0]) else None
                })

        # Sort by percentage descending
        detected.sort(key=lambda x: x['percentage'], reverse=True)
        return detected

    def _check_bilingual(self, sample_values: List[str]) -> bool:
        """Check if values contain both Chinese and English characters."""
        if not sample_values:
            return False

        for val in sample_values[:10]:
            val_str = str(val)
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in val_str)
            has_english = any(c.isalpha() and ord(c) < 128 for c in val_str)
            if has_chinese and has_english:
                return True
        return False

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if column contains date values."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            valid_dates = parsed.notna().sum()
            if valid_dates / len(sample) > 0.7:
                return True
        except:
            pass
        return False