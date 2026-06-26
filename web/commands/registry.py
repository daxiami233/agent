"""Slash command registry for the web runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent_runtime.providers import OpenAIProvider


@dataclass(slots=True)
class CommandContext:
    """Runtime services exposed to slash commands."""

    provider: OpenAIProvider | None
    history_path: Path
    print_error: Callable[[str], None]
    print_help: Callable[[], None]
    print_hint: Callable[[str], None]
    print_info: Callable[[str], None]
    print_model: Callable[[], None]
    print_status: Callable[[], None]
    clear_screen: Callable[[], None]


CommandHandler = Callable[[CommandContext], bool]


@dataclass(slots=True)
class SlashCommand:
    """A slash command with optional aliases."""

    name: str
    description: str
    handler: CommandHandler
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def execute(self, raw_command: str, context: CommandContext) -> bool | None:
        command_name = raw_command.strip().split(maxsplit=1)[0]
        canonical_name = self._aliases.get(command_name, command_name)
        command = self._commands.get(canonical_name)
        if command is None:
            return None
        return command.handler(context)

    def completion_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for command in sorted(self._commands.values(), key=lambda item: item.name):
            rows.append((command.name, command.description))
            rows.extend(
                (alias, f"alias for {command.name}")
                for alias in sorted(command.aliases)
            )
        return rows

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def help_rows(self) -> list[tuple[str, str]]:
        return [
            (command.name, command.description)
            for command in sorted(self._commands.values(), key=lambda item: item.name)
        ]
