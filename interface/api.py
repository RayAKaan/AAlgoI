import sys
import os
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aalgoi.core.problem_spec import ProblemSpec
from aalgoi.pipeline import UniversalSolver
from interface.nl_parser import parse_description, extract_data_from_description

_FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None
    HTTPException = None
    uvicorn = None


class SolveRequest(BaseModel):
    name: str = ""
    problem_type: str = "transformation"
    inputs: Dict = {"data": {"type": "list"}}
    outputs: Dict = {"result": {"type": "list"}}
    constraints: List[str] = []
    objectives: List[Dict] = []
    data: Any = None
    use_llm: bool = False


class SolveNLRequest(BaseModel):
    description: str
    data: Any = None
    use_llm: bool = False


class ExplainRequest(BaseModel):
    algorithm_name: str
    detail: str = "short"


def create_app(solver: Optional[UniversalSolver] = None):
    if not _FASTAPI_AVAILABLE:
        return None

    if solver is None:
        solver = UniversalSolver()

    app = FastAPI(title="AAlgoI API", version="1.0.0",
                  description="Universal Problem-Solving System API")

    @app.post("/solve")
    async def solve(request: SolveRequest):
        try:
            spec = ProblemSpec(
                name=request.name or "api_problem",
                inputs=request.inputs,
                outputs=request.outputs,
                constraints=request.constraints,
                objectives=request.objectives
            )
            data = request.data if request.data is not None else [3, 1, 2]
            result = solver.solve(spec, data, use_llm=request.use_llm)

            return {
                "success": result["success"],
                "result": _serialize_result(result["result"]),
                "time_ms": result["time_ms"],
                "pipeline": result["pipeline"],
                "strategy": result.get("selection", {}).get("synthesis_strategy", ""),
                "confidence": result.get("selection", {}).get("confidence", 0),
                "validation": result["validation"],
                "explanation": [
                    {"algorithm": e.algorithm_name, "summary": e.summary[:200]}
                    for e in result.get("explanation", [])
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/solve/nl")
    async def solve_natural_language(request: SolveNLRequest):
        try:
            spec = parse_description(request.description)
            data = request.data if request.data is not None else extract_data_from_description(request.description)
            if data is None:
                data = [3, 1, 4, 1, 5]

            result = solver.solve(spec, data, use_llm=request.use_llm)

            return {
                "success": result["success"],
                "result": _serialize_result(result["result"]),
                "time_ms": result["time_ms"],
                "pipeline": result["pipeline"],
                "strategy": result.get("selection", {}).get("synthesis_strategy", ""),
                "confidence": result.get("selection", {}).get("confidence", 0),
                "parsed_spec": {
                    "name": spec.name,
                    "problem_type": spec.problem_type.value,
                    "inputs": spec.inputs,
                    "outputs": spec.outputs
                },
                "validation": result["validation"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def get_stats():
        try:
            stats = solver.get_stats()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/explain")
    async def explain_algorithm(request: ExplainRequest):
        from aalgoi.core.explainer import Explainer
        explainer = Explainer()
        exp = explainer.explain(request.algorithm_name, detail=request.detail)
        return {
            "algorithm": exp.algorithm_name,
            "summary": exp.summary,
            "complexity": exp.complexity,
            "steps": exp.steps,
            "best_for": exp.best_for
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "aalgoi-api"}

    return app


def _serialize_result(result: Any) -> Any:
    if isinstance(result, (int, float, str, bool)):
        return result
    if isinstance(result, (list, tuple)):
        return [_serialize_result(r) for r in result]
    if isinstance(result, dict):
        return {str(k): _serialize_result(v) for k, v in result.items()}
    return str(result)


def run(host="0.0.0.0", port=8000, solver=None):
    if not _FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    app = create_app(solver)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
