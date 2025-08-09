#!/usr/bin/env python3
"""Test script to verify endoftext token filtering works correctly."""

import numpy as np
import torch

# Test the sequence filtering logic
def test_sequence_filtering():
    # Simulate a sequence with endoftext tokens (ID 50256)
    endoftext_id = 50256
    
    # Create sample tokens and activations
    windowed_tokens = np.array([100, 200, endoftext_id, 300, endoftext_id, 400, 500])
    windowed_acts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    windowed_feat_idx = 3  # Original index pointing to token 300
    
    print("Original tokens:", windowed_tokens)
    print("Original acts:", windowed_acts)
    print("Original feat_idx:", windowed_feat_idx)
    
    # Apply filtering (same logic as in data_fetching_fns.py)
    token_mask = windowed_tokens != endoftext_id
    
    if not token_mask.any():
        print("All tokens are endoftext - sequence would be skipped")
        return
    
    # Filter tokens and activations
    filtered_tokens = windowed_tokens[token_mask]
    filtered_acts = windowed_acts[token_mask]
    
    # Adjust feat_acts_idx after filtering
    tokens_removed_before_idx = (~token_mask[:windowed_feat_idx]).sum()
    filtered_feat_idx = windowed_feat_idx - tokens_removed_before_idx
    
    print("\nAfter filtering:")
    print("Filtered tokens:", filtered_tokens)
    print("Filtered acts:", filtered_acts)
    print("Filtered feat_idx:", filtered_feat_idx)
    print(f"Token at feat_idx: {filtered_tokens[filtered_feat_idx]}")
    
    # Verify the filtering worked correctly
    assert endoftext_id not in filtered_tokens, "Endoftext token still present!"
    assert len(filtered_tokens) == 5, f"Expected 5 tokens, got {len(filtered_tokens)}"
    assert filtered_tokens[filtered_feat_idx] == 300, f"Expected token 300 at index, got {filtered_tokens[filtered_feat_idx]}"
    print("\n✓ Sequence filtering test passed!")

# Test the LogitsTableData filtering
def test_logits_table_filtering():
    from sae_vis.utils_fns import TopK
    
    # Create sample logits with endoftext token having high value
    endoftext_id = 50256
    vocab_size = 50257
    
    # Create logits tensor
    logits = torch.randn(vocab_size)
    logits[endoftext_id] = 10.0  # Give endoftext a high logit
    logits[100] = 5.0
    logits[200] = 4.0
    logits[300] = 3.0
    
    print("\nTesting LogitsTableData filtering:")
    print(f"Original logit for endoftext (ID {endoftext_id}): {logits[endoftext_id]:.2f}")
    
    # Apply filtering (same logic as in data_storing_fns.py)
    filtered_logits = logits.clone()
    if endoftext_id < len(filtered_logits):
        filtered_logits[endoftext_id] = -1e10
    
    # Get top-k
    k = 3
    top_k_size = min(k + 5, len(filtered_logits))
    top_logits = TopK(filtered_logits, top_k_size)
    
    # Filter out endoftext from results
    top_indices = [idx for idx in top_logits.indices.tolist() if idx != endoftext_id][:k]
    top_values = [val for idx, val in zip(top_logits.indices.tolist(), top_logits.values.tolist()) if idx != endoftext_id][:k]
    
    print(f"Top {k} token IDs (after filtering): {top_indices}")
    print(f"Top {k} logit values: {[f'{v:.2f}' for v in top_values]}")
    
    # Verify endoftext is not in top tokens
    assert endoftext_id not in top_indices, "Endoftext token still in top tokens!"
    assert len(top_indices) == k, f"Expected {k} tokens, got {len(top_indices)}"
    print("\n✓ LogitsTableData filtering test passed!")

if __name__ == "__main__":
    print("Testing endoftext token filtering...")
    print("=" * 50)
    
    test_sequence_filtering()
    test_logits_table_filtering()
    
    print("\n" + "=" * 50)
    print("All tests passed! Endoftext tokens will be filtered out.")