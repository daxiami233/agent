"""Built-in web slash commands."""

from .help import help_command
from .model import model_command
from .registry import CommandContext, CommandRegistry, SlashCommand
from .status import status_command

__all__ = [
    "CommandContext",
    "CommandRegistry",
    "SlashCommand",
    "help_command",
    "model_command",
    "status_command",
]
