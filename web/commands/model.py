"""The /model command."""

from .registry import CommandContext, SlashCommand


def handle_model(context: CommandContext) -> bool:
    context.print_model()
    return False


def model_command() -> SlashCommand:
    return SlashCommand(
        name="/model",
        description="显示当前模型",
        handler=handle_model,
    )
