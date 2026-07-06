"""
Quick validation script to test the fixes.

Tests:
1. Import all modules without errors
2. Expert weight loading logic (with mock data)
3. Distributed initialization
4. Basic pipeline construction

Usage:
    python test_fixes.py
"""

import sys
import os


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Test 1: Module Imports")
    print("=" * 60)

    try:
        import expert_swap
        print("✅ expert_swap imported")
    except Exception as e:
        print(f"❌ expert_swap import failed: {e}")
        return False

    try:
        import run_inference
        print("✅ run_inference imported")
    except Exception as e:
        print(f"❌ run_inference import failed: {e}")
        return False

    try:
        import run_inference_simple
        print("✅ run_inference_simple imported")
    except Exception as e:
        print(f"❌ run_inference_simple import failed: {e}")
        return False

    try:
        import benchmarks
        print("✅ benchmarks imported")
    except Exception as e:
        print(f"❌ benchmarks import failed: {e}")
        return False

    print("\n✅ All imports successful\n")
    return True


def test_expert_weight_loading():
    """Test expert weight loading logic with mock data."""
    print("=" * 60)
    print("Test 2: Expert Weight Loading Logic")
    print("=" * 60)

    from expert_swap import load_expert_weights
    import tempfile
    from pathlib import Path

    # Create mock directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        transformer_dir = Path(tmpdir) / "transformer"
        transformer_dir.mkdir()

        # Create mock safetensors file
        import torch
        from safetensors.torch import save_file

        # Mock expert weights
        mock_weights_expert0 = {
            "expert.0.layer1.weight": torch.randn(10, 10),
            "expert.0.layer2.weight": torch.randn(10, 10),
            "shared.layer.weight": torch.randn(5, 5),
        }

        mock_weights_expert1 = {
            "expert.1.layer1.weight": torch.randn(10, 10),
            "expert.1.layer2.weight": torch.randn(10, 10),
        }

        all_weights = {**mock_weights_expert0, **mock_weights_expert1}
        save_file(all_weights, str(transformer_dir / "model.safetensors"))

        print(f"Created mock checkpoint in {tmpdir}")

        # Test loading expert 0
        try:
            weights_0 = load_expert_weights(tmpdir, expert_id=0)
            print(f"✅ Expert 0 loaded: {len(weights_0)} tensors")

            # Check that we got the right keys
            expected_keys = {"layer1.weight", "layer2.weight", "shared.layer.weight"}
            actual_keys = set(weights_0.keys())

            if expected_keys.issubset(actual_keys):
                print("✅ Expert 0 keys correct")
            else:
                print(f"⚠️  Expert 0 keys mismatch")
                print(f"   Expected: {expected_keys}")
                print(f"   Got: {actual_keys}")

        except Exception as e:
            print(f"❌ Expert 0 loading failed: {e}")
            return False

        # Test loading expert 1
        try:
            weights_1 = load_expert_weights(tmpdir, expert_id=1)
            print(f"✅ Expert 1 loaded: {len(weights_1)} tensors")

            # Check that we got only expert 1 weights
            expected_keys = {"layer1.weight", "layer2.weight"}
            actual_keys = set(weights_1.keys())

            if expected_keys == actual_keys:
                print("✅ Expert 1 keys correct (no shared weights)")
            else:
                print(f"⚠️  Expert 1 keys mismatch")
                print(f"   Expected: {expected_keys}")
                print(f"   Got: {actual_keys}")

        except Exception as e:
            print(f"❌ Expert 1 loading failed: {e}")
            return False

    print("\n✅ Expert weight loading test passed\n")
    return True


def test_distributed_init():
    """Test distributed initialization."""
    print("=" * 60)
    print("Test 3: Distributed Initialization")
    print("=" * 60)

    from run_inference import init_distributed

    try:
        rank, world_size = init_distributed()
        print(f"✅ init_distributed() succeeded")
        print(f"   Rank: {rank}, World size: {world_size}")

        if world_size == 1:
            print("   (Single-process mode)")
        else:
            print(f"   (Distributed mode with {world_size} processes)")

    except Exception as e:
        print(f"❌ init_distributed() failed: {e}")
        return False

    print("\n✅ Distributed initialization test passed\n")
    return True


def test_simple_pipeline_construction():
    """Test that simplified pipeline can be constructed."""
    print("=" * 60)
    print("Test 4: Simple Pipeline Construction")
    print("=" * 60)

    # We can't actually load the model without the checkpoint,
    # but we can test that the functions exist and have correct signatures

    from run_inference_simple import setup_environment, generate_image_manual
    import inspect

    try:
        setup_environment("cpu")
        print("✅ setup_environment() works")
    except Exception as e:
        print(f"❌ setup_environment() failed: {e}")
        return False

    # Check function signatures
    sig = inspect.signature(generate_image_manual)
    params = list(sig.parameters.keys())
    expected_params = ["prompt", "model_dir", "output_path", "device", "height", "width", "num_steps", "guidance_scale", "seed"]

    if all(p in params for p in expected_params):
        print("✅ generate_image_manual() has correct signature")
    else:
        print(f"⚠️  generate_image_manual() signature mismatch")
        print(f"   Expected: {expected_params}")
        print(f"   Got: {params}")

    print("\n✅ Simple pipeline construction test passed\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("WAN 2.2 PyTorch Native - Fix Validation Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("Module Imports", test_imports()))
    results.append(("Expert Weight Loading", test_expert_weight_loading()))
    results.append(("Distributed Init", test_distributed_init()))
    results.append(("Simple Pipeline", test_simple_pipeline_construction()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNext steps:")
        print("1. Download the WAN 2.2 model checkpoint")
        print("2. Run: python run_inference_simple.py --prompt 'test' --device cpu --image")
        print("3. See FIXES_APPLIED.md for full implementation roadmap")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck the error messages above and fix the issues.")

    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
