"""Strict, closed action schemas independent of the planner implementation."""

from dataclasses import asdict, dataclass
import json
from .models import canonical_id

ACTION_ARGUMENTS = {
    "SEARCH_PAPERS": "query", "EXPAND_REFERENCES": "paper_id", "EXPAND_CITATIONS": "paper_id",
    "EXPLORE_TOPIC": "topic_id", "INSPECT_PAPER": "paper_id", "STOP_RETRIEVAL": None,
}

# A prompt/observability revision, not a change to action validation or memory.
RELIABILITY_VERSION = "controlled-reliability-v2-contract"
CONTROLLED_TOOL_CONTRACT = (
    "Acquisition tool contract: INSPECT_PAPER fetches metadata for an unseen cited or related target of an eligible observed paper. "
    "Papers in top_papers or observed_metadata already have metadata loaded. INSPECT_PAPER cannot refresh them, reveal a longer abstract, or fetch full text. "
    "EXPAND_REFERENCES and EXPAND_CITATIONS use observed papers satisfying the relevance and path-depth guards. "
    "Each acquisition action with the same arguments may complete only once; repeating it is invalid, not pagination. "
    "EXPLORE_TOPIC uses a topic observed on an eligible paper. SEARCH_PAPERS searches the stated retrieval environment. "
    "STOP_RETRIEVAL ends acquisition. Current protected proposals are validated eligible examples, not an exhaustive list. "
    "Recent action history may be incomplete; absence from it does not establish that an operation has not occurred. "
    "Archive reads return at most 512 characters, using character offsets into a serialized selected observation, not a full paper or full abstract. "
    "Reads and context edits consume the same model-call ceiling as decisions; reads do not execute acquisition actions. "
)


@dataclass(frozen=True)
class Action:
    action: str
    arguments: dict
    reason: str = ""

    def __post_init__(self) -> None:
        name, arguments, reason = self.action, self.arguments, self.reason
        if not isinstance(name, str) or name not in ACTION_ARGUMENTS or not isinstance(arguments, dict):
            raise ValueError("Unknown action or malformed arguments")
        argument = ACTION_ARGUMENTS[name]
        if set(arguments) != ({argument} if argument else set()):
            raise ValueError("Action arguments do not match schema")
        if not isinstance(reason, str) or len(reason) > 2000:
            raise ValueError("Action reason must be text of at most 2000 characters")
        arguments = dict(arguments)
        if argument:
            target = arguments[argument]
            if not isinstance(target, str) or not target.strip():
                raise ValueError("Action target must be nonempty text")
            if argument == "query":
                if len(target) > 1000:
                    raise ValueError("Search query exceeds 1000 characters")
                arguments[argument] = " ".join(target.split())
            else:
                arguments[argument] = canonical_id(target, "T" if argument == "topic_id" else "W")
        object.__setattr__(self, "arguments", arguments)

    @property
    def signature(self) -> str:
        arguments = dict(self.arguments)
        if "query" in arguments:
            arguments["query"] = " ".join(arguments["query"].lower().split())
        return json.dumps([self.action, arguments], sort_keys=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def parse(cls, value: dict) -> "Action":
        if not isinstance(value, dict) or set(value) - {"action", "arguments", "reason"}:
            raise ValueError("Action must be an object with action, arguments, and optional reason")
        return cls(value.get("action"), value.get("arguments"), value.get("reason", ""))


def make_action(name: str, value: str | None = None, reason: str = "") -> Action:
    if name not in ACTION_ARGUMENTS:
        raise ValueError("Unknown action")
    argument = ACTION_ARGUMENTS[name]
    return Action.parse({"action": name, "arguments": {argument: value} if argument else {}, "reason": reason})


def action_schema(permitted: list[str] | None = None) -> dict:
    """Closed JSON schemas for the same action space enforced by Action.parse."""
    names = list(ACTION_ARGUMENTS) if permitted is None else permitted
    if not names or any(name not in ACTION_ARGUMENTS for name in names):
        raise ValueError("Unknown action in schema")
    variants = []
    for name in names:
        argument = ACTION_ARGUMENTS[name]
        properties = {argument: {"type": "string", "minLength": 1,
                                  "maxLength": 1000 if argument == "query" else 100}} if argument else {}
        variants.append({"type": "object", "additionalProperties": False, "required": ["action", "arguments"],
            "properties": {"action": {"const": name}, "arguments": {"type": "object", "additionalProperties": False,
                "properties": properties, "required": list(properties)}, "reason": {"type": "string", "maxLength": 2000}}})
    return {"oneOf": variants}
