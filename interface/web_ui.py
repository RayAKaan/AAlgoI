import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ast
from typing import Any, Optional
from aalgoi.core.problem_spec import ProblemSpec
from aalgoi.pipeline import UniversalSolver
from interface.nl_parser import parse_description, extract_data_from_description


def _parse_data_input(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, (list, dict, int, float, str)):
            return parsed
        return stripped
    except (ValueError, SyntaxError):
        return stripped

_GRADIO_AVAILABLE = False
try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None


def create_interface(solver: Optional[UniversalSolver] = None):
    if not _GRADIO_AVAILABLE:
        return None

    if solver is None:
        solver = UniversalSolver()

    def solve_from_nl(description: str, data_input: str):
        if not description.strip():
            return "Please describe a problem.", "", "", "", ""

        spec = parse_description(description)
        data = extract_data_from_description(description)

        parsed = _parse_data_input(data_input)
        if parsed is not None:
            data = parsed

        if data is None:
            data = [3, 1, 4, 1, 5]

        if isinstance(data, list) and all(isinstance(x, int) for x in data):
            pass

        result = solver.solve(spec, data)

        output_str = str(result.get("result", ""))
        if isinstance(result.get("result"), list) and len(result["result"]) > 20:
            output_str = str(result["result"][:20]) + f" ... ({len(result['result'])} total)"

        strategy = result.get("selection", {}).get("synthesis_strategy", "unknown")
        confidence = result.get("selection", {}).get("confidence", 0)
        time_str = f"{result.get('time_ms', 0):.2f} ms"
        pipeline_str = " → ".join(result.get("pipeline", []))

        validation_info = ""
        for v in result.get("validation", []):
            status = "✓" if v.get("passed") else "✗"
            validation_info += f"{status} {v['algorithm']}\n"

        explanation = ""
        for exp in result.get("explanation", []):
            explanation += f"**{exp.algorithm_name}**: {exp.summary[:120]}...\n\n"

        return output_str, f"{strategy} (conf: {confidence:.2f})", time_str, pipeline_str, validation_info, explanation

    def solve_from_spec_json(spec_json: str, data_input: str):
        import json
        try:
            spec_dict = json.loads(spec_json)
            spec = ProblemSpec.from_dict(spec_dict)
        except Exception as e:
            return f"Invalid JSON: {e}", "", "", "", "", ""

        data = _parse_data_input(data_input) or [3, 1, 2]

        result = solver.solve(spec, data)
        output_str = str(result.get("result", ""))
        strategy = result.get("selection", {}).get("synthesis_strategy", "unknown")
        confidence = result.get("selection", {}).get("confidence", 0)
        time_str = f"{result.get('time_ms', 0):.2f} ms"
        pipeline_str = " → ".join(result.get("pipeline", []))
        validation_info = ""
        for v in result.get("validation", []):
            status = "✓" if v.get("passed") else "✗"
            validation_info += f"{status} {v['algorithm']}\n"
        explanation = ""
        for exp in result.get("explanation", []):
            explanation += f"**{exp.algorithm_name}**: {exp.summary[:120]}...\n\n"
        return output_str, strategy, time_str, pipeline_str, validation_info, explanation

    with gr.Blocks(title="AAlgoI Universal Solver", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AAlgoI — Universal Problem Solver")
        gr.Markdown("Describe a problem in natural language, or provide a JSON ProblemSpec.")

        with gr.Tab("Natural Language"):
            with gr.Row():
                with gr.Column(scale=2):
                    nl_input = gr.Textbox(
                        label="Describe your problem",
                        placeholder="e.g., sort these numbers: 3, 1, 4, 1, 5",
                        lines=3
                    )
                    data_input_nl = gr.Textbox(
                        label="Data (optional, override extracted)",
                        placeholder="[3, 1, 4, 1, 5]",
                        lines=1
                    )
                    run_btn_nl = gr.Button("Solve", variant="primary")

                with gr.Column(scale=2):
                    output_nl = gr.Textbox(label="Result", lines=3)
                    strategy_nl = gr.Textbox(label="Strategy", lines=1)
                    time_nl = gr.Textbox(label="Time", lines=1)
                    pipeline_nl = gr.Textbox(label="Pipeline", lines=1)

            with gr.Row():
                with gr.Column():
                    validation_nl = gr.Textbox(label="Validation", lines=3)
                with gr.Column():
                    explanation_nl = gr.Markdown(label="Explanation")

            run_btn_nl.click(
                solve_from_nl,
                inputs=[nl_input, data_input_nl],
                outputs=[output_nl, strategy_nl, time_nl, pipeline_nl, validation_nl, explanation_nl]
            )

        with gr.Tab("JSON ProblemSpec"):
            with gr.Row():
                with gr.Column(scale=2):
                    spec_json = gr.Textbox(
                        label="ProblemSpec JSON",
                        placeholder='{"name": "sort", "inputs": {"data": {"type": "list[int]"}}, "outputs": {"sorted": {"type": "list[int]"}}}',
                        lines=5
                    )
                    data_input_json = gr.Textbox(
                        label="Data",
                        placeholder="[3, 1, 4, 1, 5]",
                        lines=1
                    )
                    run_btn_json = gr.Button("Solve", variant="primary")

                with gr.Column(scale=2):
                    output_json = gr.Textbox(label="Result", lines=3)
                    strategy_json = gr.Textbox(label="Strategy", lines=1)
                    time_json = gr.Textbox(label="Time", lines=1)
                    pipeline_json = gr.Textbox(label="Pipeline", lines=1)

            with gr.Row():
                with gr.Column():
                    validation_json = gr.Textbox(label="Validation", lines=3)
                with gr.Column():
                    explanation_json = gr.Markdown(label="Explanation")

            run_btn_json.click(
                solve_from_spec_json,
                inputs=[spec_json, data_input_json],
                outputs=[output_json, strategy_json, time_json, pipeline_json, validation_json, explanation_json]
            )

        gr.Markdown("### About\nAAlgoI is a universal problem-solving system that analyzes, synthesizes, and executes algorithms for any computational problem.")

    return demo


def launch(solver=None, share=False, server_port=7860):
    demo = create_interface(solver)
    if demo is None:
        print("Gradio not available. Install with: pip install gradio")
        return
    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch()
