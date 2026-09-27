"""Durable export registry for opted-in, recoverable finalization only."""
from agent_base.media_backend.flush import RegistryEntry


class ConfigExportRegistry:
    def __init__(self, agent):
        self.agent = agent

    async def load(self, agent_uuid):
        raw = self.agent.agent_config.extras.get("export_registry")
        if raw is not None:
            return {path: RegistryEntry(**value) for path, value in raw.items()}
        # Existing conversations already keep successful export metadata. Seed
        # from it without republishing unchanged siblings on the first opt-in.
        return {
            m.extras["export_path"]: RegistryEntry(m.extras["export_path"], m.extras["blake3_hash"], m.media_id)
            for m in self.agent.agent_config.media_registry.values()
            if m.extras.get("export_path") and m.extras.get("blake3_hash")
        }

    async def save(self, agent_uuid, registry):
        from dataclasses import asdict
        self.agent.agent_config.extras["export_registry"] = {path: asdict(value) for path, value in registry.items()}
