#!/usr/bin/env python3
"""Test script to verify our imports are working correctly."""

import sys
from pathlib import Path

# Add our local crosscoder-vis to the path
sys.path.insert(0, str(Path(__file__).parent / "crosscoder-vis"))

try:
    from sae_vis.data_config_classes import SaeVisConfig
    print("✓ Successfully imported SaeVisConfig")
    
    from sae_vis.data_storing_fns import SaeVisData
    print("✓ Successfully imported SaeVisData")
    
    from sae_vis.model_fns import CrossCoderConfig, CrossCoder
    print("✓ Successfully imported CrossCoderConfig and CrossCoder")
    
    from sae_vis.data_fetching_fns import get_feature_data
    print("✓ Successfully imported get_feature_data")
    
    # Test that we're using the right version
    import sae_vis.data_fetching_fns as dff
    print(f"✓ Using data_fetching_fns from: {dff.__file__}")
    
    # Check if get_feature_data expects LanguageModel
    import inspect
    sig = inspect.signature(dff.get_feature_data)
    print(f"✓ get_feature_data signature: {sig}")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()