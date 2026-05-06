"""Schema version constants. Bump when a driver schema changes."""

CONFIG_SCHEMA_VERSION = "1.0"
GRAPH_FSM_SCHEMA_VERSION = "1.0"
TYPED_SCORE_RECORD_SCHEMA_VERSION = "1.0"
ABLATION_FLAGS_SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR = 1


def check_supported(observed: str, name: str) -> None:
    """Raise ValueError if observed major != SUPPORTED_MAJOR."""
    major = int(observed.split(".")[0])
    if major != SUPPORTED_MAJOR:
        raise ValueError(
            f"{name} schema_version {observed} unsupported; expected major {SUPPORTED_MAJOR}"
        )
