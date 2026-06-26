"""The /status command."""

from .registry import CommandContext, SlashCommand


def handle_status(context: CommandContext) -> bool:
    context.print_status()
    return False


def status_command() -> SlashCommand:
    return SlashCommand(
        name="/status",
        description="显示会话状态",
        handler=handle_status,
    )
