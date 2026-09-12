"""Stable target selectors for Qwen3.5 component experiments."""

COMPONENT_TARGETS: dict[str, tuple[str, ...]] = {
    "deltanet": ("re:.*linear_attn.*",),
    "attention": ("re:.*self_attn.*",),
    "ffn": ("re:.*mlp.*",),
    "vision": ("re:.*visual.*",),
}


def targets_for(components: tuple[str, ...]) -> tuple[str, ...]:
    targets: list[str] = []
    for component in components:
        try:
            targets.extend(COMPONENT_TARGETS[component])
        except KeyError as error:
            allowed = ", ".join(sorted(COMPONENT_TARGETS))
            raise ValueError(
                f"Unknown component {component!r}; choose one of: {allowed}"
            ) from error
    return tuple(dict.fromkeys(targets))
