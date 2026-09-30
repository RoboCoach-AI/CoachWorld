__all__ = ["WanModelAction", "CausalWanModel", "Wan2_2_VAE", "FlowMatchScheduler"]


def __getattr__(name):
    if name == "WanModelAction":
        from coachworld.wan.model import WanModelAction

        return WanModelAction
    if name == "CausalWanModel":
        from coachworld.wan.causal_model import CausalWanModel

        return CausalWanModel
    if name == "Wan2_2_VAE":
        from coachworld.wan.vae import Wan2_2_VAE

        return Wan2_2_VAE
    if name == "FlowMatchScheduler":
        from coachworld.wan.scheduler import FlowMatchScheduler

        return FlowMatchScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
