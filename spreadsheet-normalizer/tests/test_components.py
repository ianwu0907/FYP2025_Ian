"""
Simple test script to verify the pipeline works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import SpreadsheetEncoder
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_encoder():
    """Test the encoder module."""
    logger.info("Testing Encoder...")

    # Create a simple test dataframe
    df = pd.DataFrame({
        'Year/年份': [2020, 2021, 2022],
        'Category/类别': ['A', 'B', 'C'],
        'Value/数值': [100, 200, 300]
    })

    config = {'anchor_neighborhood': 2}
    encoder = SpreadsheetEncoder(config)

    # Test encoding
    result = encoder.encode(df)

    assert 'encoded_text' in result
    assert 'metadata' in result
    assert result['original_shape'] == (3, 3)

    logger.info("✓ Encoder test passed")
    logger.info(f"  Compression ratio: {result['metadata']['compression_ratio']:.2f}x")
    return result


def test_components():
    """Test all components."""
    logger.info("\n" + "="*60)
    logger.info("TESTING SPREADSHEET NORMALIZER COMPONENTS")
    logger.info("="*60 + "\n")

    # Test Encoder
    try:
        encoded_data = test_encoder()
        logger.info("✓ Test 1: Encoder - PASSED\n")
    except Exception as e:
        logger.error(f"✗ Test 1: Encoder - FAILED: {e}\n")
        return False

    logger.info("="*60)
    logger.info("COMPONENT TESTS COMPLETED")
    logger.info("="*60 + "\n")

    return True


if __name__ == '__main__':
    success = test_components()
    sys.exit(0 if success else 1)