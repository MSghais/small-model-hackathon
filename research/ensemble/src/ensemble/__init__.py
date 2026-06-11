"""Research ensemble package: JEPA and world-model tracks."""

__all__ = ["Ensemble", "WorldEnsemble"]


def __getattr__(name: str):
    if name == "Ensemble":
        from ensemble.jepa_ensemble import Ensemble

        return Ensemble
    if name == "WorldEnsemble":
        from ensemble.world_ensemble import WorldEnsemble

        return WorldEnsemble
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
