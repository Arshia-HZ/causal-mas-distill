"""
Script 03: Build datasets for each selector.

Emits one dataset per selector, using the unified template format.
"""

import argparse
import json
from pathlib import Path

from src.debate.schema import Trace
from src.selection.causal import CausalSelector
from src.selection.random_lenmatched import RandomLenMatchedSelector
from src.selection.confidence import ConfidenceSelector
from src.selection.prm import PRMSelector
from src.selection.oracle_filter import OracleFilterSelector
from src.selection.last_round_only import LastRoundOnlySelector
from src.render.template import TemplateRenderer


SELECTORS = {
    "causal": CausalSelector,
    "random_lenmatched": RandomLenMatchedSelector,
    "confidence": ConfidenceSelector,
    "prm": PRMSelector,
    "oracle_filter": OracleFilterSelector,
    "last_round_only": LastRoundOnlySelector,
}


def main():
    parser = argparse.ArgumentParser(description="Build datasets per selector")
    parser.add_argument("--traces", type=str, required=True, help="Path to traces JSON")
    parser.add_argument("--utilities", type=str, required=True, help="Path to utilities JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--token-budget", type=int, default=100000, help="Token budget per selector")
    args = parser.parse_args()

    # Load traces
    with open(args.traces) as f:
        traces_data = json.load(f)
    traces = [Trace.from_dict(t) for t in traces_data]

    # Load utilities
    with open(args.utilities) as f:
        utilities = json.load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset for each selector
    for selector_name, selector_class in SELECTORS.items():
        print(f"Building dataset for selector: {selector_name}")
        
        selector = selector_class()
        selected_mids = selector.select(traces, utilities, args.token_budget)
        
        # Render selected messages
        examples = TemplateRenderer.render_for_sft(traces, selected_mids)
        
        # Save dataset
        output_path = output_dir / f"{selector_name}_dataset.json"
        with open(output_path, "w") as f:
            json.dump(examples, f, indent=2)
        
        print(f"  Selected {len(selected_mids)} messages, {len(examples)} examples")
        print(f"  Saved to {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()