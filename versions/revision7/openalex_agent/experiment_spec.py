"""One serializable benchmark specification for CLI and dashboard adapters."""

from dataclasses import asdict, dataclass, field

from .context_planner import InferenceLimits
from .controller import Budget


@dataclass(frozen=True)
class BenchmarkSpec:
    corpus: str
    questions: str
    output: str
    suite: str = "controlled-five"
    methods: tuple[str, ...] | None = None
    action_budgets: tuple[int, ...] = (5,)
    trials: int = 1
    trial_seeds: tuple[int, ...] | None = None
    budget: Budget = field(default_factory=Budget)
    inference_limits: InferenceLimits = field(default_factory=InferenceLimits)
    labels: str | None = None
    factual_questions: str | None = None
    allow_deterministic: bool = False
    ablations: bool = False
    require_evidence: bool = False

    def backend_kwargs(self):
        return dict(output=self.output, suite=self.suite, methods=self.methods, action_budgets=self.action_budgets,
            trials=self.trials, trial_seeds=self.trial_seeds, budget=self.budget, inference_limits=self.inference_limits,
            labels_path=self.labels, factual_questions_path=self.factual_questions, allow_deterministic=self.allow_deterministic,
            ablations=self.ablations, require_evidence=self.require_evidence)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_cli(cls, args):
        from .cli import run_budget
        return cls(corpus=args.corpus, questions=args.questions, output=args.output, suite=args.suite,
            methods=tuple(args.methods) if args.methods else None, action_budgets=tuple(args.action_budgets), trials=args.trials,
            trial_seeds=tuple(args.trial_seeds) if args.trial_seeds is not None else None,
            budget=run_budget(args), inference_limits=InferenceLimits(args.max_total_tokens,args.max_llm_cost_usd,
                args.active_context_tokens,args.input_tokens), labels=args.labels,factual_questions=args.factual_questions,
            allow_deterministic=args.allow_deterministic,ablations=args.ablations,require_evidence=args.require_evidence)
