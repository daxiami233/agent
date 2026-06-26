"""The /help command."""

from .registry import CommandContext, SlashCommand


def handle_help(context: CommandContext) -> bool:
    context.print_help()
    return False


def help_command() -> SlashCommand:
    return SlashCommand(
        name="/help",
        description="显示命令",
        handler=handle_help,
    )
