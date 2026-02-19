"""Pipeline validation and evaluation script.

Note: This validates final output only. The modular pipeline doesn't save
intermediate results for efficiency - it processes everything in memory.
"""

from pathlib import Path

import yaml


def count_images(path_obj: Path) -> int:
    """Count PNG images in directory and subdirectories."""
    return sum(1 for p in path_obj.rglob("*.png") if p.is_file())


def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    original_path = Path(config["paths"]["original_raw"])
    final_path = Path(config["paths"]["ad_filtered"])

    if not original_path.exists():
        print(f"ERROR: Original directory not found: {original_path}")
        return

    if not final_path.exists():
        print(f"ERROR: Final directory not found: {final_path}")
        print("Run the pipeline first: make run")
        return

    original_count = count_images(original_path)
    final_count = count_images(final_path)
    retention = (final_count / original_count * 100) if original_count > 0 else 0

    print(f"\nPipeline Results:")
    print(f"  {original_count} → {final_count} images ({retention:.1f}% retained)")
    print(f"  {original_count - final_count} images filtered")

    final_images = {f.name for f in final_path.glob("*.png")}

    # Known children test
    print(f"\nKnown children check (age < 13):")
    known_children = ["crop (524).png"]

    children_passed = 0
    for child in known_children:
        if child not in final_images:
            print(f"  [PASS] {child}")
            children_passed += 1
        else:
            print(f"  [FAIL] {child} (should be filtered)")

    # Known mannequins test
    print(f"\nKnown mannequins check:")
    known_mannequins = ["crop (829).png", "crop (910).png"]

    mannequins_passed = 0
    for mannequin in known_mannequins:
        if mannequin not in final_images:
            print(f"  [PASS] {mannequin}")
            mannequins_passed += 1
        else:
            print(f"  [FAIL] {mannequin} (should be filtered)")

    # Known ads test
    print(f"\nKnown ads check:")
    known_ads = ["crop (63).png", "crop (990).png"]

    ads_passed = 0
    for known_ad in known_ads:
        if known_ad not in final_images:
            print(f"  [PASS] {known_ad}")
            ads_passed += 1
        else:
            print(f"  [FAIL] {known_ad} (should be filtered)")

    # Sanity checks
    print(f"\nSanity checks:")
    checks = []

    final_list = list(final_path.glob("*.png"))
    final_unique = len({f.name for f in final_list})
    checks.append(("No duplicates", len(final_list) == final_unique))
    checks.append(("Final ≤ original", final_count <= original_count))

    original_images = {f.name for f in original_path.glob("*.png")}
    checks.append(("All from original", final_images.issubset(original_images)))

    for name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    all_passed = (
        all(r for _, r in checks)
        and children_passed == len(known_children)
        and mannequins_passed == len(known_mannequins)
        and ads_passed == len(known_ads)
    )
    print(f"\n{'All checks passed' if all_passed else 'Some checks failed'}")


if __name__ == "__main__":
    main()
