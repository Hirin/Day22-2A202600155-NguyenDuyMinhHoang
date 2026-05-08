import sys
import importlib

STEPS = {
    1: ("01_langsmith_rag_pipeline", "LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing", "Prompt Hub A/B Routing"),
    3: ("03_ragas_evaluation", "RAGAS Evaluation"),
    4: ("04_guardrails_validator", "Guardrails Validators"),
}


def run_step(step_num: int):
    if step_num not in STEPS:
        print(f"Unknown step: {step_num}. Choose from {list(STEPS.keys())}")
        return
    module_name, description = STEPS[step_num]
    print(f"\n{'=' * 60}")
    print(f"  Running Step {step_num}: {description}")
    print(f"{'=' * 60}\n")
    mod = importlib.import_module(module_name)
    mod.main()


def main():
    if len(sys.argv) > 1 and sys.argv[1].startswith("--step"):
        step_num = int(sys.argv[2])
        run_step(step_num)
    else:
        for step_num in sorted(STEPS.keys()):
            try:
                run_step(step_num)
            except Exception as e:
                print(f"Step {step_num} failed: {e}")
                continue
        print("\nAll steps complete!")


if __name__ == "__main__":
    main()
