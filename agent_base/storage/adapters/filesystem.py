"""Filesystem storage adapters.

These adapters store data as JSON files in a directory structure:
    {base_path}/
        agent_config/
            {agent_uuid}.json
        conversation_history/
            {agent_uuid}/
                001.json
                002.json
                index.json
        agent_runs/
            {agent_uuid}/
                {run_id}.jsonl
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from ..base import (
    AgentConfig,
    AgentConfigAdapter,
    Conversation,
    ConversationAdapter,
    AgentRunAdapter,
    LogEntry,
)
from ..serialization import (
    serialize_config,
    deserialize_config,
    serialize_conversation,
    deserialize_conversation,
    serialize_log_entry,
    deserialize_log_entry,
)
from ..exceptions import StorageOperationError
from ...logging import get_logger

logger = get_logger(__name__)


def _json_default(obj: Any) -> str:
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class FilesystemAgentConfigAdapter(AgentConfigAdapter):
    """Filesystem adapter for agent configuration.

    Stores configs in: {base_path}/agent_config/{agent_uuid}.json
    """

    def __init__(self, base_path: str = "./data"):
        """Initialize filesystem agent config adapter.

        Args:
            base_path: Root directory for storage (default: ./data)
        """
        self.base_path = Path(base_path)
        self._config_dir = self.base_path / "agent_config"

    async def connect(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._config_dir.mkdir(exist_ok=True)

    async def save(self, config: AgentConfig) -> None:
        """Save agent configuration to JSON file with atomic write."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        config_file = self._config_dir / f"{config.agent_uuid}.json"
        temp_file = config_file.with_suffix(".tmp")

        data = serialize_config(config)
        content = json.dumps(data, indent=2, default=_json_default)

        async with aiofiles.open(temp_file, "w") as f:
            await f.write(content)

        # Atomic rename
        temp_file.replace(config_file)

        logger.debug(
            "Saved agent config",
            agent_uuid=config.agent_uuid,
            backend="filesystem"
        )

    async def load(self, agent_uuid: str) -> AgentConfig | None:
        """Load agent configuration from JSON file."""
        config_file = self._config_dir / f"{agent_uuid}.json"

        if not config_file.exists():
            return None

        async with aiofiles.open(config_file, "r") as f:
            content = await f.read()

        data = json.loads(content)
        config = deserialize_config(data)

        logger.debug(
            "Loaded agent config",
            agent_uuid=agent_uuid,
            backend="filesystem"
        )
        return config

    async def delete(self, agent_uuid: str) -> bool:
        """Delete agent configuration file."""
        config_file = self._config_dir / f"{agent_uuid}.json"

        if not config_file.exists():
            return False

        await aiofiles.os.remove(config_file)
        logger.debug(
            "Deleted agent config",
            agent_uuid=agent_uuid,
            backend="filesystem"
        )
        return True

    async def update_title(self, agent_uuid: str, title: str) -> bool:
        """Update the title for an agent session."""
        config = await self.load(agent_uuid)
        if config is None:
            return False

        config.title = title
        await self.save(config)

        logger.debug(
            "Updated agent title",
            agent_uuid=agent_uuid,
            title=title,
            backend="filesystem"
        )
        return True

    async def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict], int]:
        """List all agent sessions with metadata."""
        if not self._config_dir.exists():
            return [], 0

        sessions: list[dict] = []

        config_files = list(self._config_dir.glob("*.json"))

        for config_file in config_files:
            if config_file.name.endswith(".tmp"):
                continue

            try:
                async with aiofiles.open(config_file, "r") as f:
                    content = await f.read()
                data = json.loads(content)

                # Use file modification time as fallback for updated_at
                stat = config_file.stat()
                file_mtime = datetime.fromtimestamp(stat.st_mtime)

                sessions.append({
                    "agent_uuid": data.get("agent_uuid", config_file.stem),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at") or file_mtime.isoformat(),
                    "total_runs": data.get("total_runs", 0),
                })
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Failed to load config file",
                    file=str(config_file),
                    error=str(e)
                )
                continue

        # Sort by updated_at descending (newest first)
        sessions.sort(
            key=lambda x: x.get("updated_at") or "",
            reverse=True
        )

        total = len(sessions)
        paginated = sessions[offset:offset + limit]

        logger.debug(
            "Listed agent sessions",
            count=len(paginated),
            total=total,
            backend="filesystem"
        )
        return paginated, total


class FilesystemConversationAdapter(ConversationAdapter):
    """Filesystem adapter for conversation history.

    Stores conversations in: {base_path}/conversation_history/{agent_uuid}/{seq:03d}.json
    Maintains an index.json for sequence tracking.
    """

    def __init__(self, base_path: str = "./data"):
        """Initialize filesystem conversation adapter.

        Args:
            base_path: Root directory for storage (default: ./data)
        """
        self.base_path = Path(base_path)
        self._conv_dir = self.base_path / "conversation_history"

    async def connect(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._conv_dir.mkdir(exist_ok=True)

    async def save(self, conversation: Conversation) -> None:
        """Save conversation with automatic sequence numbering."""
        agent_dir = self._conv_dir / conversation.agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Get next sequence number from index
        index_file = agent_dir / "index.json"
        if index_file.exists():
            async with aiofiles.open(index_file, "r") as f:
                content = await f.read()
            index = json.loads(content)
            last_sequence = index.get("last_sequence", 0)
        else:
            last_sequence = 0

        next_sequence = last_sequence + 1
        conversation.sequence_number = next_sequence

        # Save conversation with padded sequence number
        conv_file = agent_dir / f"{next_sequence:03d}.json"
        data = serialize_conversation(conversation)
        content = json.dumps(data, indent=2, default=_json_default)

        async with aiofiles.open(conv_file, "w") as f:
            await f.write(content)

        # Update index
        index = {
            "last_sequence": next_sequence,
            "total_conversations": next_sequence,
            "updated_at": datetime.now().isoformat()
        }
        async with aiofiles.open(index_file, "w") as f:
            await f.write(json.dumps(index, indent=2))

        logger.debug(
            "Saved conversation",
            agent_uuid=conversation.agent_uuid,
            sequence_number=next_sequence,
            backend="filesystem"
        )

    async def load_history(
        self,
        agent_uuid: str,
        limit: int = 20,
        offset: int = 0
    ) -> list[Conversation]:
        """Load paginated conversation history (newest first)."""
        agent_dir = self._conv_dir / agent_uuid

        if not agent_dir.exists():
            return []

        # Get all conversation files sorted by sequence number descending
        conv_files = sorted(agent_dir.glob("[0-9]*.json"), reverse=True)

        # Apply pagination
        paginated = conv_files[offset:offset + limit]

        # Load conversations
        conversations = []
        for conv_file in paginated:
            async with aiofiles.open(conv_file, "r") as f:
                content = await f.read()
            data = json.loads(content)
            conversations.append(deserialize_conversation(data))

        logger.debug(
            "Loaded conversation history",
            agent_uuid=agent_uuid,
            count=len(conversations),
            backend="filesystem"
        )
        return conversations

    async def load_cursor(
        self,
        agent_uuid: str,
        before: int | None = None,
        limit: int = 20
    ) -> tuple[list[Conversation], bool]:
        """Load conversations with cursor-based pagination (newest first)."""
        agent_dir = self._conv_dir / agent_uuid

        if not agent_dir.exists():
            return [], False

        # Get all conversation files sorted descending
        conv_files = sorted(agent_dir.glob("[0-9]*.json"), reverse=True)

        # Filter by cursor if provided
        if before is not None:
            conv_files = [
                f for f in conv_files
                if int(f.stem) < before
            ]

        # Take limit + 1 to determine has_more
        selected = conv_files[:limit + 1]
        has_more = len(selected) > limit
        files_to_load = selected[:limit]

        # Load conversations
        conversations = []
        for conv_file in files_to_load:
            async with aiofiles.open(conv_file, "r") as f:
                content = await f.read()
            data = json.loads(content)
            conversations.append(deserialize_conversation(data))

        logger.debug(
            "Loaded conversation cursor",
            agent_uuid=agent_uuid,
            count=len(conversations),
            has_more=has_more,
            backend="filesystem"
        )
        return conversations, has_more


class FilesystemAgentRunAdapter(AgentRunAdapter):
    """Filesystem adapter for agent run logs.

    Stores logs in JSONL format: {base_path}/agent_runs/{agent_uuid}/{run_id}.jsonl
    """

    def __init__(self, base_path: str = "./data"):
        """Initialize filesystem agent run adapter.

        Args:
            base_path: Root directory for storage (default: ./data)
        """
        self.base_path = Path(base_path)
        self._runs_dir = self.base_path / "agent_runs"

    async def connect(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(exist_ok=True)

    async def save_logs(
        self,
        agent_uuid: str,
        run_id: str,
        logs: list[LogEntry]
    ) -> None:
        """Save agent run logs in JSONL format."""
        agent_dir = self._runs_dir / agent_uuid
        agent_dir.mkdir(parents=True, exist_ok=True)

        log_file = agent_dir / f"{run_id}.jsonl"

        # Build JSONL content
        lines = [
            json.dumps(serialize_log_entry(entry), default=_json_default)
            for entry in logs
        ]
        content = "\n".join(lines)
        if content:
            content += "\n"

        async with aiofiles.open(log_file, "w") as f:
            await f.write(content)

        logger.debug(
            "Saved agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="filesystem"
        )

    async def load_logs(
        self,
        agent_uuid: str,
        run_id: str
    ) -> list[LogEntry]:
        """Load agent run logs from JSONL file."""
        log_file = self._runs_dir / agent_uuid / f"{run_id}.jsonl"

        if not log_file.exists():
            return []

        logs = []
        async with aiofiles.open(log_file, "r") as f:
            content = await f.read()

        for line in content.splitlines():
            if line.strip():
                data = json.loads(line)
                logs.append(deserialize_log_entry(data))

        logger.debug(
            "Loaded agent run logs",
            agent_uuid=agent_uuid,
            run_id=run_id,
            count=len(logs),
            backend="filesystem"
        )
        return logs
