# Licensing boundaries and outstanding owner decision

The repository has **no project-wide license grant** for its original code. Making it publicly visible does not imply unrestricted reuse or make all components permissively licensed. The owner has not selected a blanket license in this preparation.

|Component|Boundary|Recorded terms|
|---|---|---|
|Original research/application code, adapters, UI, tests and documentation|Root files and `openalex_agent/`, excluding identified third-party components; frozen original code under `versions/revision7/`|No explicit project-wide reuse license supplied. Authorship and provenance remain intact; a license is an owner decision|
|Official Meta context-language-models leaves|`openalex_agent/_vendor/meta_clm/` and `versions/revision7/openalex_agent/_vendor/meta_clm/`|CC BY-NC 4.0; full LICENSE, NOTICE and revision/file provenance remain in each copy. Noncommercial restrictions are not removed by a future license on original code|
|OpenAI tokenizer ranks and study encoding attribution|`openalex_agent/tokenizer_data/` and its frozen R7 counterpart|Accompanying MIT LICENSE and README identify the public tiktoken ranks, source and checksum|
|mini-swe-agent and Harbor references|Meta's NOTICE in both vendor copies|Upstream attributions are retained. The NOTICE describes adapted upstream harness conventions; the public package vendors selected context/budget leaves, not the Harbor harness. Harbor is not vendored here|
|Installed dependencies|`pyproject.toml`, requirements and locks|Each dependency has its own upstream terms; installation does not relicense those packages|
|Public example/metadata fixtures|`tests/fixtures/`, `benchmarks/`|Synthetic versus authentic metadata provenance is documented separately. A future original-code license must not be assumed to grant rights in third-party scholarly works or full text|

The custom provider/environment adapters are outside the Meta vendor boundary; their location is not a license grant. Frozen R7's source manifest identifies historical bytes rather than replacing any upstream terms.

Outstanding decisions: the owner should choose and document intended reuse terms for original work, including third-party exceptions, and assess compatibility for the intended distribution/use. Until then, describe this as publicly viewable research source with component-specific terms, not an entirely MIT-licensed or unrestricted open-source distribution. No licenses or notices have been changed or weakened. This inventory is not legal advice or a full dependency-license compatibility opinion.
