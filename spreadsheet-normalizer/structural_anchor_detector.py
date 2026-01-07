"""
Structural Anchor Detector Module - SpreadsheetLLM Implementation

This module implements the structural anchor detection algorithm described in:
"SPREADSHEETLLM: Encoding Spreadsheets for Large Language Models" (Appendix C)

Key concepts:
- Structural anchors are heterogeneous rows/columns at table boundaries
- Used to compress spreadsheets while preserving layout information
- Filters out ~75% of content while preserving 97% of boundary rows/columns
"""

from __future__ import annotations
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AnchorInfo:
    """Information about a structural anchor (row or column)."""
    index: int
    anchor_type: str  # 'row' or 'column'
    heterogeneity_score: float
    features: Dict[str, Any]


@dataclass
class BoundaryCandidate:
    """Candidate table boundary."""
    top: int
    left: int
    bottom: int
    right: int
    score: float
    has_header: bool
    reasons: List[str]


class StructuralAnchorDetector:
    """
    Detects structural anchors in spreadsheets following SpreadsheetLLM methodology.
    
    Algorithm steps (from Appendix C):
    1. Enumerate bounding lines by finding discrepancies in neighboring rows/columns
    2. Compose candidate boundaries using any two rows and two columns
    3. Filter unreasonable boundaries using heuristics
    4. Handle overlapping boundaries
    5. Extract structural anchors from remaining boundaries
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the detector.
        
        Args:
            config: Configuration dictionary with optional keys:
                - k_neighborhood: Number of cells to keep around anchors (default: 4)
                - min_table_size: Minimum table size (default: 2x2)
                - heterogeneity_threshold: Threshold for considering row/col heterogeneous
        """
        config = config or {}
        self.k = config.get('k_neighborhood', 4)
        self.min_table_rows = config.get('min_table_rows', 2)
        self.min_table_cols = config.get('min_table_cols', 2)
        self.heterogeneity_threshold = config.get('heterogeneity_threshold', 0.3)
        
        logger.info(
            f"Initialized StructuralAnchorDetector with k={self.k}, "
            f"min_table={self.min_table_rows}x{self.min_table_cols}"
        )
    
    def detect_anchors(self, 
                       df: pd.DataFrame, 
                       format_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect structural anchors in a spreadsheet.
        
        Args:
            df: Input DataFrame
            format_info: Optional format information (colors, borders, merges, etc.)
        
        Returns:
            Dictionary containing:
                - row_anchors: List of anchor row indices
                - column_anchors: List of anchor column indices
                - boundary_candidates: List of detected table boundaries
                - extraction_mask: Boolean mask indicating which cells to keep
        """
        logger.info(f"Starting structural anchor detection on {df.shape[0]}x{df.shape[1]} spreadsheet")
        
        # Step 1: Enumerate bounding lines (heterogeneous rows/columns)
        row_anchors = self._detect_heterogeneous_rows(df, format_info)
        col_anchors = self._detect_heterogeneous_columns(df, format_info)
        
        logger.info(f"Found {len(row_anchors)} row anchors and {len(col_anchors)} column anchors")
        
        # Step 2: Compose candidate boundaries
        boundary_candidates = self._compose_boundary_candidates(
            row_anchors, col_anchors, df.shape
        )
        
        logger.info(f"Generated {len(boundary_candidates)} boundary candidates")
        
        # Step 3: Filter unreasonable boundaries
        filtered_boundaries = self._filter_boundaries(boundary_candidates, df, format_info)
        
        logger.info(f"After filtering: {len(filtered_boundaries)} boundaries remain")
        
        # Step 4: Handle overlapping boundaries
        final_boundaries = self._resolve_overlaps(filtered_boundaries, df)
        
        logger.info(f"After overlap resolution: {len(final_boundaries)} final boundaries")
        
        # Step 5: Extract anchors from final boundaries
        final_row_anchors, final_col_anchors = self._extract_boundary_anchors(
            final_boundaries
        )
        
        # Step 6: Create extraction mask (keep k-neighborhood around anchors)
        extraction_mask = self._create_extraction_mask(
            df.shape, final_row_anchors, final_col_anchors
        )
        
        result = {
            'row_anchors': sorted(final_row_anchors),
            'column_anchors': sorted(final_col_anchors),
            'boundary_candidates': [self._boundary_to_dict(b) for b in final_boundaries],
            'extraction_mask': extraction_mask,
            'statistics': {
                'original_rows': df.shape[0],
                'original_cols': df.shape[1],
                'kept_rows': extraction_mask.any(axis=1).sum(),
                'kept_cols': extraction_mask.any(axis=0).sum(),
                'compression_ratio': (1 - extraction_mask.sum() / extraction_mask.size) * 100
            }
        }
        
        logger.info(
            f"Extraction complete: keeping {result['statistics']['kept_rows']}/"
            f"{df.shape[0]} rows, {result['statistics']['kept_cols']}/{df.shape[1]} cols "
            f"({result['statistics']['compression_ratio']:.1f}% compression)"
        )
        
        return result
    
    def _detect_heterogeneous_rows(self, 
                                    df: pd.DataFrame,
                                    format_info: Optional[Dict[str, Any]] = None) -> List[AnchorInfo]:
        """
        Detect heterogeneous rows that likely represent table boundaries.
        
        A row is heterogeneous if it differs significantly from its neighbors in:
        - Cell values (text vs numbers, empty vs filled)
        - Formatting (colors, borders, fonts)
        - Merged cells
        """
        anchors = []
        
        for row_idx in range(len(df)):
            score = self._calculate_row_heterogeneity(row_idx, df, format_info)
            
            if score >= self.heterogeneity_threshold:
                features = self._extract_row_features(row_idx, df, format_info)
                anchors.append(AnchorInfo(
                    index=row_idx,
                    anchor_type='row',
                    heterogeneity_score=score,
                    features=features
                ))
        
        return anchors
    
    def _detect_heterogeneous_columns(self,
                                       df: pd.DataFrame,
                                       format_info: Optional[Dict[str, Any]] = None) -> List[AnchorInfo]:
        """
        Detect heterogeneous columns that likely represent table boundaries.
        """
        anchors = []
        
        for col_idx in range(len(df.columns)):
            score = self._calculate_column_heterogeneity(col_idx, df, format_info)
            
            if score >= self.heterogeneity_threshold:
                features = self._extract_column_features(col_idx, df, format_info)
                anchors.append(AnchorInfo(
                    index=col_idx,
                    anchor_type='column',
                    heterogeneity_score=score,
                    features=features
                ))
        
        return anchors
    
    def _calculate_row_heterogeneity(self,
                                      row_idx: int,
                                      df: pd.DataFrame,
                                      format_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate heterogeneity score for a row by comparing with neighbors.
        
        Higher score = more different from neighbors = likely boundary
        """
        scores = []
        
        # Compare with previous row
        if row_idx > 0:
            scores.append(self._compare_rows(row_idx - 1, row_idx, df, format_info))
        
        # Compare with next row
        if row_idx < len(df) - 1:
            scores.append(self._compare_rows(row_idx, row_idx + 1, df, format_info))
        
        return max(scores) if scores else 0.0
    
    def _calculate_column_heterogeneity(self,
                                         col_idx: int,
                                         df: pd.DataFrame,
                                         format_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate heterogeneity score for a column.
        """
        scores = []
        
        # Compare with previous column
        if col_idx > 0:
            scores.append(self._compare_columns(col_idx - 1, col_idx, df, format_info))
        
        # Compare with next column
        if col_idx < len(df.columns) - 1:
            scores.append(self._compare_columns(col_idx, col_idx + 1, df, format_info))
        
        return max(scores) if scores else 0.0
    
    def _compare_rows(self,
                      row1_idx: int,
                      row2_idx: int,
                      df: pd.DataFrame,
                      format_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Compare two adjacent rows and return dissimilarity score (0-1).
        
        Compares:
        1. Value types (text/numeric/empty)
        2. Fill patterns (empty vs filled cells)
        3. Formatting (if available)
        """
        row1 = df.iloc[row1_idx]
        row2 = df.iloc[row2_idx]
        
        differences = 0
        total_comparisons = 0
        
        # Value-based comparison
        for val1, val2 in zip(row1, row2):
            total_comparisons += 1
            
            # Check if both are empty
            val1_empty = pd.isna(val1) or (isinstance(val1, str) and val1.strip() == '')
            val2_empty = pd.isna(val2) or (isinstance(val2, str) and val2.strip() == '')
            
            if val1_empty != val2_empty:
                differences += 1
                continue
            
            # If both non-empty, check type mismatch
            if not val1_empty and not val2_empty:
                type1 = 'numeric' if isinstance(val1, (int, float)) else 'text'
                type2 = 'numeric' if isinstance(val2, (int, float)) else 'text'
                
                if type1 != type2:
                    differences += 0.5
        
        # Format-based comparison (if available)
        if format_info and 'row_formats' in format_info:
            format1 = format_info['row_formats'].get(row1_idx, {})
            format2 = format_info['row_formats'].get(row2_idx, {})
            
            # Compare colors
            if format1.get('colors') != format2.get('colors'):
                differences += len(df.columns) * 0.2
            
            # Compare borders
            if format1.get('has_border') != format2.get('has_border'):
                differences += len(df.columns) * 0.2
        
        return min(differences / max(total_comparisons, 1), 1.0)
    
    def _compare_columns(self,
                         col1_idx: int,
                         col2_idx: int,
                         df: pd.DataFrame,
                         format_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Compare two adjacent columns and return dissimilarity score.
        """
        col1 = df.iloc[:, col1_idx]
        col2 = df.iloc[:, col2_idx]
        
        differences = 0
        total_comparisons = 0
        
        for val1, val2 in zip(col1, col2):
            total_comparisons += 1
            
            val1_empty = pd.isna(val1) or (isinstance(val1, str) and val1.strip() == '')
            val2_empty = pd.isna(val2) or (isinstance(val2, str) and val2.strip() == '')
            
            if val1_empty != val2_empty:
                differences += 1
                continue
            
            if not val1_empty and not val2_empty:
                type1 = 'numeric' if isinstance(val1, (int, float)) else 'text'
                type2 = 'numeric' if isinstance(val2, (int, float)) else 'text'
                
                if type1 != type2:
                    differences += 0.5
        
        return min(differences / max(total_comparisons, 1), 1.0)
    
    def _extract_row_features(self,
                               row_idx: int,
                               df: pd.DataFrame,
                               format_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract features from a row for later analysis."""
        row = df.iloc[row_idx]
        
        non_empty = row.notna().sum()
        numeric_count = sum(1 for v in row if isinstance(v, (int, float)) and pd.notna(v))
        text_count = sum(1 for v in row if isinstance(v, str) and v.strip())
        
        features = {
            'non_empty_cells': non_empty,
            'empty_cells': len(row) - non_empty,
            'numeric_cells': numeric_count,
            'text_cells': text_count,
            'fill_ratio': non_empty / len(row) if len(row) > 0 else 0
        }
        
        # Add format features if available
        if format_info and 'row_formats' in format_info:
            row_format = format_info['row_formats'].get(row_idx, {})
            features.update({
                'has_color': bool(row_format.get('colors')),
                'has_border': bool(row_format.get('has_border')),
                'bold_cells': row_format.get('bold', 0)
            })
        
        return features
    
    def _extract_column_features(self,
                                  col_idx: int,
                                  df: pd.DataFrame,
                                  format_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract features from a column."""
        col = df.iloc[:, col_idx]
        
        non_empty = col.notna().sum()
        numeric_count = sum(1 for v in col if isinstance(v, (int, float)) and pd.notna(v))
        text_count = sum(1 for v in col if isinstance(v, str) and v.strip())
        
        return {
            'non_empty_cells': non_empty,
            'empty_cells': len(col) - non_empty,
            'numeric_cells': numeric_count,
            'text_cells': text_count,
            'fill_ratio': non_empty / len(col) if len(col) > 0 else 0
        }
    
    def _compose_boundary_candidates(self,
                                      row_anchors: List[AnchorInfo],
                                      col_anchors: List[AnchorInfo],
                                      shape: Tuple[int, int]) -> List[BoundaryCandidate]:
        """
        Compose candidate table boundaries using pairs of rows and columns.
        
        For efficiency, we limit combinations to nearby anchors.
        """
        candidates = []
        row_indices = [a.index for a in row_anchors]
        col_indices = [a.index for a in col_anchors]
        
        # Add first/last rows and columns as implicit anchors
        if 0 not in row_indices:
            row_indices = [0] + row_indices
        if shape[0] - 1 not in row_indices:
            row_indices.append(shape[0] - 1)
        
        if 0 not in col_indices:
            col_indices = [0] + col_indices
        if shape[1] - 1 not in col_indices:
            col_indices.append(shape[1] - 1)
        
        # Compose candidates from anchor pairs
        for i, top in enumerate(row_indices[:-1]):
            for bottom in row_indices[i+1:]:
                # Skip if too far apart or too close
                if bottom - top < self.min_table_rows:
                    continue
                if bottom - top > shape[0] * 0.9:  # Skip nearly full-sheet tables
                    continue
                
                for j, left in enumerate(col_indices[:-1]):
                    for right in col_indices[j+1:]:
                        if right - left < self.min_table_cols:
                            continue
                        if right - left > shape[1] * 0.9:
                            continue
                        
                        candidates.append(BoundaryCandidate(
                            top=top,
                            left=left,
                            bottom=bottom,
                            right=right,
                            score=0.0,
                            has_header=False,
                            reasons=[]
                        ))
        
        return candidates
    
    def _filter_boundaries(self,
                           candidates: List[BoundaryCandidate],
                           df: pd.DataFrame,
                           format_info: Optional[Dict[str, Any]] = None) -> List[BoundaryCandidate]:
        """
        Filter unreasonable boundary candidates using heuristics.
        
        Heuristics from Appendix C:
        1. Check sparsity (proportion of numbers/text in rows and columns)
        2. Check internal region integrity
        3. Check for header-like rows
        4. Check table size reasonability
        """
        filtered = []
        
        for candidate in candidates:
            score, reasons = self._score_boundary(candidate, df, format_info)
            
            if score > 0:  # Threshold for keeping
                candidate.score = score
                candidate.reasons = reasons
                filtered.append(candidate)
        
        # Sort by score
        filtered.sort(key=lambda x: x.score, reverse=True)
        
        return filtered
    
    def _score_boundary(self,
                        candidate: BoundaryCandidate,
                        df: pd.DataFrame,
                        format_info: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
        """
        Score a boundary candidate based on table likelihood.
        
        Returns:
            (score, reasons) where score > 0 means likely a table
        """
        score = 0.0
        reasons = []
        
        # Extract region
        region = df.iloc[candidate.top:candidate.bottom+1, candidate.left:candidate.right+1]
        
        # Check 1: Size
        height = candidate.bottom - candidate.top + 1
        width = candidate.right - candidate.left + 1
        
        if height < self.min_table_rows or width < self.min_table_cols:
            return 0.0, ["Too small"]
        
        score += min(height * width / 100, 10)  # Size bonus (capped)
        
        # Check 2: Non-empty ratio
        non_empty_ratio = region.notna().sum().sum() / region.size
        if non_empty_ratio < 0.1:
            return 0.0, ["Too sparse"]
        
        score += non_empty_ratio * 10
        reasons.append(f"Density: {non_empty_ratio:.1%}")
        
        # Check 3: Header detection
        first_row = region.iloc[0]
        first_row_text = sum(1 for v in first_row if isinstance(v, str))
        first_row_numeric = sum(1 for v in first_row if isinstance(v, (int, float)) and pd.notna(v))
        
        if first_row_text >= width * 0.5:  # Mostly text
            score += 5
            candidate.has_header = True
            reasons.append("Has text header")
        
        # Check 4: Data consistency
        # Data rows should have consistent types
        if height > 1:
            data_rows = region.iloc[1:]
            numeric_cols = 0
            
            for col_idx in range(width):
                col = data_rows.iloc[:, col_idx]
                numeric_ratio = sum(1 for v in col if isinstance(v, (int, float)) and pd.notna(v)) / len(col)
                if numeric_ratio > 0.7:
                    numeric_cols += 1
            
            if numeric_cols > 0:
                score += numeric_cols * 2
                reasons.append(f"{numeric_cols} numeric columns")
        
        # Check 5: Format clues (if available)
        if format_info and 'row_formats' in format_info:
            top_format = format_info['row_formats'].get(candidate.top, {})
            if top_format.get('has_border') or top_format.get('colors'):
                score += 3
                reasons.append("Top row has formatting")
        
        return score, reasons
    
    def _resolve_overlaps(self,
                          boundaries: List[BoundaryCandidate],
                          df: pd.DataFrame) -> List[BoundaryCandidate]:
        """
        Resolve overlapping boundary candidates.
        
        From Appendix C Figure 6: Handle common overlap patterns by choosing
        the boundary with stronger evidence (higher score, better headers, etc.)
        """
        if not boundaries:
            return []
        
        final = []
        used_regions = []
        
        for candidate in boundaries:
            region = (candidate.top, candidate.left, candidate.bottom, candidate.right)
            
            # Check if overlaps with any accepted region
            overlaps = False
            for used_region in used_regions:
                if self._regions_overlap(region, used_region):
                    overlaps = True
                    break
            
            if not overlaps:
                final.append(candidate)
                used_regions.append(region)
        
        return final
    
    def _regions_overlap(self,
                         region1: Tuple[int, int, int, int],
                         region2: Tuple[int, int, int, int]) -> bool:
        """Check if two regions overlap."""
        top1, left1, bottom1, right1 = region1
        top2, left2, bottom2, right2 = region2
        
        # No overlap if one is completely to the left/right/above/below the other
        if bottom1 < top2 or bottom2 < top1:
            return False
        if right1 < left2 or right2 < left1:
            return False
        
        return True
    
    def _extract_boundary_anchors(self,
                                   boundaries: List[BoundaryCandidate]) -> Tuple[Set[int], Set[int]]:
        """Extract row and column anchors from final boundaries."""
        row_anchors = set()
        col_anchors = set()
        
        for b in boundaries:
            row_anchors.add(b.top)
            row_anchors.add(b.bottom)
            col_anchors.add(b.left)
            col_anchors.add(b.right)
        
        return row_anchors, col_anchors
    
    def _create_extraction_mask(self,
                                 shape: Tuple[int, int],
                                 row_anchors: Set[int],
                                 col_anchors: Set[int]) -> np.ndarray:
        """
        Create boolean mask indicating which cells to keep.
        
        Keep cells within k distance from any anchor.
        """
        mask = np.zeros(shape, dtype=bool)
        
        for row_anchor in row_anchors:
            for row in range(max(0, row_anchor - self.k), 
                            min(shape[0], row_anchor + self.k + 1)):
                mask[row, :] = True
        
        for col_anchor in col_anchors:
            for col in range(max(0, col_anchor - self.k),
                            min(shape[1], col_anchor + self.k + 1)):
                mask[:, col] = True
        
        return mask
    
    def _boundary_to_dict(self, boundary: BoundaryCandidate) -> Dict[str, Any]:
        """Convert BoundaryCandidate to dictionary for JSON serialization."""
        return {
            'top': boundary.top,
            'left': boundary.left,
            'bottom': boundary.bottom,
            'right': boundary.right,
            'score': boundary.score,
            'has_header': boundary.has_header,
            'reasons': boundary.reasons
        }
    
    def extract_compressed_sheet(self,
                                  df: pd.DataFrame,
                                  extraction_mask: np.ndarray) -> pd.DataFrame:
        """
        Extract compressed spreadsheet based on mask.
        
        Returns a new DataFrame with only the rows/columns marked in the mask.
        """
        # Find rows and columns to keep
        rows_to_keep = extraction_mask.any(axis=1)
        cols_to_keep = extraction_mask.any(axis=0)
        
        # Extract
        compressed = df.loc[rows_to_keep, cols_to_keep]
        
        logger.info(
            f"Compressed {df.shape} -> {compressed.shape} "
            f"({(1 - compressed.size/df.size)*100:.1f}% reduction)"
        )
        
        return compressed


def demo_usage():
    """Demonstration of structural anchor detection."""
    # Create sample spreadsheet
    data = {
        'A': ['', 'Product', 'Apple', 'Banana', 'Orange', '', 'Total', ''],
        'B': ['', 'Q1', '100', '150', '200', '', '450', ''],
        'C': ['', 'Q2', '120', '180', '220', '', '520', ''],
        'D': ['', 'Q3', '110', '160', '210', '', '480', ''],
    }
    df = pd.DataFrame(data)
    
    print("Original Spreadsheet:")
    print(df)
    print()
    
    # Detect anchors
    detector = StructuralAnchorDetector(config={'k_neighborhood': 2})
    result = detector.detect_anchors(df)
    
    print("Detection Results:")
    print(f"Row anchors: {result['row_anchors']}")
    print(f"Column anchors: {result['column_anchors']}")
    print(f"Detected {len(result['boundary_candidates'])} table(s)")
    print()
    
    # Extract compressed sheet
    compressed = detector.extract_compressed_sheet(df, result['extraction_mask'])
    print("Compressed Spreadsheet:")
    print(compressed)
    print()
    print(f"Statistics: {result['statistics']}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo_usage()
