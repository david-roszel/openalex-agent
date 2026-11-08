"""Trusted local tokenizer subprocess for the isolated SDK's pre-send hook."""
import json
import sys

from .context_memory import serialized
from .token_budget import count


def main():
    value=json.load(sys.stdin)
    tokens=count(serialized(value['body']))
    print(json.dumps(dict(tokens=tokens,allowed=tokens<=value['limit'])))


if __name__=='__main__':main()
