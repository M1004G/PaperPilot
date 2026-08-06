"""Tests for codegen_agent.py. llm_client is monkeypatched -- no real LLM calls."""
import json

from backend import codegen_agent
from backend.ingestion_agent import IngestedPaper, Section


def make_paper(method_text="We use a transformer encoder trained with Adam."):
    return IngestedPaper(
        title="Test Paper", abstract="An abstract about a model.",
        sections=[Section(heading="Methodology", text=method_text)],
        full_text=f"An abstract about a model.\n\n{method_text}",
    )


GOOD_PAPER_INFO = {
    "title": "Test Paper", "task": "image classification", "framework": "pytorch",
    "datasets": [{"name": "CIFAR-10", "source": "torchvision", "splits": "train/test", "preprocessing": ""}],
    "model": {"architecture": "ResNet-18", "base_model": "", "key_components": "", "input_shape": "", "output_shape": ""},
    "training": {"optimizer": "Adam", "learning_rate": "1e-3", "batch_size": "32", "epochs": "10", "loss_function": "cross entropy"},
    "evaluation": {"metrics": ["accuracy"], "reported_results": "95%"},
    "gaps": ["Weight decay not specified"],
}


class TestExtractPaperInfo:
    def test_parses_valid_response(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps(GOOD_PAPER_INFO))
        info = codegen_agent.extract_paper_info(make_paper())
        assert info["title"] == "Test Paper"
        assert info["training"]["optimizer"] == "Adam"
        assert info["gaps"] == ["Weight decay not specified"]

    def test_invalid_json_falls_back_gracefully(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        info = codegen_agent.extract_paper_info(make_paper())
        assert info["title"] == "Test Paper"  # falls back to paper.title
        assert info["gaps"]  # non-empty explanatory gap

    def test_missing_gaps_key_defaults_to_empty_list(self, fake_llm, monkeypatch):
        info_no_gaps = {k: v for k, v in GOOD_PAPER_INFO.items() if k != "gaps"}
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps(info_no_gaps))
        info = codegen_agent.extract_paper_info(make_paper())
        assert info["gaps"] == []

    def test_prioritizes_method_section_over_full_text(self, fake_llm, monkeypatch):
        captured = {}
        def fake_complete(system, prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps(GOOD_PAPER_INFO)
        monkeypatch.setattr(fake_llm, "complete_json", fake_complete)
        paper = IngestedPaper(
            title="T", abstract="Abstract text.",
            sections=[
                Section(heading="Introduction", text="Irrelevant intro filler." * 50),
                Section(heading="Methodology", text="THE ACTUAL METHOD DESCRIPTION"),
            ],
            full_text="Abstract text.\n" + ("Irrelevant intro filler." * 50) + "\nTHE ACTUAL METHOD DESCRIPTION",
        )
        codegen_agent.extract_paper_info(paper)
        assert "THE ACTUAL METHOD DESCRIPTION" in captured["prompt"]


class TestPlanFiles:
    def test_parses_valid_plan(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [
                {"filename": "model.py", "purpose": "the model", "depends_on": []},
                {"filename": "train.py", "purpose": "training loop", "depends_on": ["model.py"]},
            ]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert [f["filename"] for f in plan] == ["model.py", "train.py"]
        assert plan[1]["depends_on"] == ["model.py"]

    def test_drops_dependency_on_unknown_file(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [{"filename": "model.py", "purpose": "the model", "depends_on": ["nonexistent.py"]}]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert plan[0]["depends_on"] == []

    def test_falls_back_to_default_plan_on_invalid_json(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: "not json")
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert plan == codegen_agent._DEFAULT_PLAN

    def test_falls_back_to_default_plan_on_empty_files_list(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({"files": []}))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert plan == codegen_agent._DEFAULT_PLAN

    def test_caps_at_max_files(self, fake_llm, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "CODEGEN_MAX_FILES", 2)
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [{"filename": f"f{i}.py", "purpose": "x", "depends_on": []} for i in range(5)]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert len(plan) == 2


class TestTopoOrder:
    def test_orders_dependencies_first(self):
        plan = [
            {"filename": "train.py", "purpose": "", "depends_on": ["model.py", "dataset.py"]},
            {"filename": "model.py", "purpose": "", "depends_on": []},
            {"filename": "dataset.py", "purpose": "", "depends_on": []},
        ]
        ordered = codegen_agent._topo_order(plan)
        names = [f["filename"] for f in ordered]
        assert names.index("model.py") < names.index("train.py")
        assert names.index("dataset.py") < names.index("train.py")

    def test_falls_back_to_declared_order_on_cycle(self):
        plan = [
            {"filename": "a.py", "purpose": "", "depends_on": ["b.py"]},
            {"filename": "b.py", "purpose": "", "depends_on": ["a.py"]},
        ]
        ordered = codegen_agent._topo_order(plan)
        assert ordered == plan

    def test_independent_files_in_deterministic_order(self):
        plan = [
            {"filename": "z.py", "purpose": "", "depends_on": []},
            {"filename": "a.py", "purpose": "", "depends_on": []},
        ]
        ordered = codegen_agent._topo_order(plan)
        assert [f["filename"] for f in ordered] == ["a.py", "z.py"]


class TestGenerateCodeFiles:
    def test_generates_one_file_per_plan_entry(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [
            {"filename": "model.py", "purpose": "the model", "depends_on": []},
            {"filename": "train.py", "purpose": "training", "depends_on": ["model.py"]},
        ])
        outputs = iter(["class Model:\n    def forward(self, x):\n        return x", "from model import Model\n\ndef train():\n    pass\n\ntrain()"])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: next(outputs))
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert files == {
            "model.py": "class Model:\n    def forward(self, x):\n        return x",
            "train.py": "from model import Model\n\ndef train():\n    pass\n\ntrain()",
        }

    def test_strips_markdown_fences(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: "```python\nclass Model:\n    def forward(self, x):\n        return x\n```")
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert files["model.py"] == "class Model:\n    def forward(self, x):\n        return x"

    def test_dependency_content_is_passed_to_dependent_file_prompt(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [
            {"filename": "model.py", "purpose": "the model", "depends_on": []},
            {"filename": "train.py", "purpose": "training", "depends_on": ["model.py"]},
        ])
        prompts = []
        def fake_complete(system, prompt, **kwargs):
            prompts.append(prompt)
            return ("class ResNetModel:\n    def forward(self, x):\n        return x" if len(prompts) == 1 else "def train():\n    pass\n\ntrain()")
        monkeypatch.setattr(fake_llm, "complete", fake_complete)
        codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "class ResNetModel" in prompts[1]  # train.py's prompt saw model.py's real content

    def test_a_failed_file_is_skipped_not_fatal(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [
            {"filename": "model.py", "purpose": "x", "depends_on": []},
            {"filename": "train.py", "purpose": "x", "depends_on": []},
        ])
        def flaky_complete(system, prompt, **kwargs):
            if "model.py" in prompt:
                raise RuntimeError("LLM call failed")
            return "def train():\n    pass\n\ntrain()"
        monkeypatch.setattr(fake_llm, "complete", flaky_complete)
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "model.py" not in files
        assert files["train.py"] == "def train():\n    pass\n\ntrain()"


class TestBuildGapReport:
    def test_lists_gaps(self):
        report = codegen_agent.build_gap_report({"gaps": ["Batch size not specified", "No seed given"]})
        assert "Batch size not specified" in report
        assert "No seed given" in report

    def test_no_gaps_message(self):
        report = codegen_agent.build_gap_report({"gaps": []})
        assert "unusually complete" in report.lower()


class TestAnalyze:
    def test_full_pipeline(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps(GOOD_PAPER_INFO))
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: "import torch\nimport torch.nn as nn")
        result = codegen_agent.analyze(make_paper())
        assert result["paper_info"]["title"] == "Test Paper"
        assert result["files"] == {"model.py": "import torch\nimport torch.nn as nn"}
        assert "gap_report" in result

    def test_codegen_disabled_skips_generation(self, fake_llm, monkeypatch):
        from backend import config
        monkeypatch.setattr(config, "CODEGEN_ENABLED", False)
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps(GOOD_PAPER_INFO))
        result = codegen_agent.analyze(make_paper())
        assert result["files"] == {}
        assert result["paper_info"]["title"] == "Test Paper"


class TestSanitizeFilename:
    def test_accepts_normal_filenames(self):
        assert codegen_agent.sanitize_filename("model.py") == "model.py"
        assert codegen_agent.sanitize_filename("requirements.txt") == "requirements.txt"
        assert codegen_agent.sanitize_filename("README.md") == "README.md"

    def test_rejects_path_traversal(self):
        assert codegen_agent.sanitize_filename("../../etc/passwd") is None
        assert codegen_agent.sanitize_filename("..") is None
        assert codegen_agent.sanitize_filename(".") is None

    def test_rejects_nested_paths(self):
        assert codegen_agent.sanitize_filename("src/model.py") is None
        assert codegen_agent.sanitize_filename("a\\b.py") is None

    def test_rejects_absolute_paths(self):
        assert codegen_agent.sanitize_filename("/etc/passwd") is None
        assert codegen_agent.sanitize_filename("C:\\Windows\\System32\\evil.py") is None

    def test_rejects_hidden_files(self):
        assert codegen_agent.sanitize_filename(".env") is None
        assert codegen_agent.sanitize_filename(".bashrc") is None

    def test_rejects_non_string_or_empty(self):
        assert codegen_agent.sanitize_filename(None) is None
        assert codegen_agent.sanitize_filename(123) is None
        assert codegen_agent.sanitize_filename("") is None
        assert codegen_agent.sanitize_filename("   ") is None

    def test_rejects_null_byte(self):
        assert codegen_agent.sanitize_filename("model.py\x00.txt") is None


class TestPlanFilesSanitizationAndDedup:
    def test_path_traversal_filename_is_dropped_from_plan(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [
                {"filename": "model.py", "purpose": "the model", "depends_on": []},
                {"filename": "../../etc/passwd", "purpose": "malicious", "depends_on": []},
            ]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert [f["filename"] for f in plan] == ["model.py"]

    def test_nested_path_filename_is_dropped_from_plan(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [{"filename": "src/model.py", "purpose": "the model", "depends_on": []}]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert plan == codegen_agent._DEFAULT_PLAN  # nothing valid survived -> falls back

    def test_duplicate_filenames_deduped_keeping_first(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [
                {"filename": "model.py", "purpose": "first version", "depends_on": []},
                {"filename": "model.py", "purpose": "duplicate version", "depends_on": []},
            ]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        assert len(plan) == 1
        assert plan[0]["purpose"] == "first version"

    def test_depends_on_cannot_reference_a_dropped_unsafe_filename(self, fake_llm, monkeypatch):
        monkeypatch.setattr(fake_llm, "complete_json", lambda *a, **k: json.dumps({
            "files": [
                {"filename": "model.py", "purpose": "the model", "depends_on": []},
                {"filename": "train.py", "purpose": "training", "depends_on": ["model.py", "../evil.py"]},
            ]
        }))
        plan = codegen_agent.plan_files(GOOD_PAPER_INFO)
        train = next(f for f in plan if f["filename"] == "train.py")
        assert train["depends_on"] == ["model.py"]


class TestSyntaxValidationAndRetry:
    def test_valid_python_is_accepted_without_retry(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        calls = []
        def fake_complete(system, prompt, **kwargs):
            calls.append(prompt)
            return "class Model:\n    def forward(self, x):\n        return x"
        monkeypatch.setattr(fake_llm, "complete", fake_complete)
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "model.py" in files
        assert len(calls) == 1  # no retry needed

    def test_syntax_error_triggers_one_retry_that_succeeds(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        responses = iter([
            "def broken(:\n    pass",  # invalid syntax
            "def fixed():\n    pass\n\nfixed()",  # valid on retry
        ])
        prompts = []
        def fake_complete(system, prompt, **kwargs):
            prompts.append(prompt)
            return next(responses)
        monkeypatch.setattr(fake_llm, "complete", fake_complete)
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert files["model.py"] == "def fixed():\n    pass\n\nfixed()"
        assert len(prompts) == 2
        assert "SYNTAX ERROR" in prompts[1]
        assert "def broken(:" in prompts[1]  # retry prompt includes the previous broken attempt

    def test_syntax_error_still_broken_after_retry_is_dropped(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: "def broken(:\n    pass")
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "model.py" not in files

    def test_retry_call_itself_failing_is_handled(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        call_count = {"n": 0}
        def fake_complete(system, prompt, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "def broken(:\n    pass"
            raise RuntimeError("retry call failed")
        monkeypatch.setattr(fake_llm, "complete", fake_complete)
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "model.py" not in files

    def test_non_python_files_are_not_syntax_checked(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "requirements.txt", "purpose": "deps", "depends_on": []}])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: "this is not { valid python syntax at all (((")
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "requirements.txt" in files  # never run through compile()


class TestRefusalDetection:
    def test_looks_like_refusal_or_empty_flags_short_content(self):
        assert codegen_agent._looks_like_refusal_or_empty("short") is True

    def test_looks_like_refusal_or_empty_flags_refusal_phrase(self):
        assert codegen_agent._looks_like_refusal_or_empty("I cannot help with generating this specific code for you.") is True
        assert codegen_agent._looks_like_refusal_or_empty("I'm sorry, but I am not able to complete this request today.") is True

    def test_looks_like_refusal_or_empty_accepts_real_code(self):
        assert codegen_agent._looks_like_refusal_or_empty("class Model:\n    def forward(self, x):\n        return x") is False

    def test_does_not_false_positive_on_code_containing_refusal_words_mid_file(self):
        # "cannot" appearing inside real code (not as the opening) shouldn't trip the check
        code = "def validate(x):\n    if x is None:\n        raise ValueError('cannot proceed with None input')\n    return x"
        assert codegen_agent._looks_like_refusal_or_empty(code) is False

    def test_refusal_response_is_dropped_without_retry(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "model.py", "purpose": "x", "depends_on": []}])
        calls = []
        def fake_complete(system, prompt, **kwargs):
            calls.append(prompt)
            return "I'm sorry, I cannot generate this code for you."
        monkeypatch.setattr(fake_llm, "complete", fake_complete)
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "model.py" not in files
        assert len(calls) == 1  # refusal isn't retried, unlike a syntax error

    def test_too_short_response_is_dropped(self, fake_llm, monkeypatch):
        monkeypatch.setattr(codegen_agent, "plan_files", lambda paper_info: [{"filename": "README.md", "purpose": "x", "depends_on": []}])
        monkeypatch.setattr(fake_llm, "complete", lambda *a, **k: "n/a")
        files = codegen_agent.generate_code_files(GOOD_PAPER_INFO)
        assert "README.md" not in files
