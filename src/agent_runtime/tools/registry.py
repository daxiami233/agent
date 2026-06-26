"""Tool specifications and dispatch registry."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints


ToolHandler = Callable[[dict[str, Any]], Any]


class ToolRegistryError(RuntimeError):
    """Base error raised by the tool registry."""


class ToolAlreadyRegisteredError(ToolRegistryError):
    """Raised when a tool name is registered more than once."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested tool is not registered."""


class ToolArgumentError(ToolRegistryError, ValueError):
    """Raised when tool arguments do not match the tool input schema."""


@dataclass(slots=True)
class ToolSpec:
    """Runtime-normalized tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    capabilities: list[str] = field(default_factory=list)

    def provider_schema(self) -> dict[str, Any]:
        """Return the function-tool shape accepted by OpenAI-compatible providers."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    """Registers and dispatches built-in and MCP tools."""

    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        self._validate_tool(tool)
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool not found: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> list[ToolSpec]:
        return [self._tools[name] for name in sorted(self._tools)]

    def provider_schemas(self) -> list[dict[str, Any]]:
        return [tool.provider_schema() for tool in self.list()]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = self.get(name)
        args = arguments or {}
        _validate_arguments(tool, args)
        return tool.handler(args)

    def _validate_tool(self, tool: ToolSpec) -> None:
        if not tool.name or not tool.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Tool name must contain only letters, numbers, '_' or '-'.")
        if not tool.description:
            raise ValueError("Tool description is required.")
        if not isinstance(tool.input_schema, dict):
            raise ValueError("Tool input_schema must be a dictionary.")


def tool_from_function(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: list[str] | None = None,
) -> ToolSpec:
    """Create a ToolSpec from a regular Python function.

    Function parameters become JSON-schema properties. Required parameters are
    those without defaults. The wrapped handler calls the function with keyword
    arguments.
    """

    signature = inspect.signature(func)
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise ValueError("Tool functions may only use keyword-compatible parameters.")
        properties[param_name] = _schema_for_annotation(
            type_hints.get(param_name, parameter.annotation)
        )
        if parameter.default is inspect.Parameter.empty:
            required.append(param_name)

    def handle(arguments: dict[str, Any]) -> Any:
        kwargs = {
            param_name: arguments[param_name]
            for param_name in properties
            if param_name in arguments
        }
        return func(**kwargs)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required

    return ToolSpec(
        name=name or func.__name__,
        description=description or inspect.getdoc(func) or f"Call {func.__name__}.",
        input_schema=schema,
        handler=handle,
        capabilities=capabilities or [],
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    capabilities: list[str] | None = None,
) -> ToolSpec | Callable[[Callable[..., Any]], ToolSpec]:
    """Decorator/function helper that turns a Python function into a ToolSpec."""

    def decorate(inner: Callable[..., Any]) -> ToolSpec:
        return tool_from_function(
            inner,
            name=name,
            description=description,
            capabilities=capabilities,
        )

    if func is None:
        return decorate
    return decorate(func)


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, UnionType}:
        schemas = [_schema_for_annotation(arg) for arg in args]
        if len(schemas) == 1:
            return schemas[0]
        if schemas:
            return {"anyOf": schemas}
        return {"type": "string"}
    if origin is Literal:
        values = list(args)
        schema = _schema_for_annotation(type(values[0])) if values else {"type": "string"}
        return {**schema, "enum": values}
    if origin is list:
        item_schema = _schema_for_annotation(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}
    if origin is None and hasattr(annotation, "__origin__"):
        origin = annotation.__origin__

    mapping = {
        type(None): {"type": "null"},
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
    }
    return dict(mapping.get(annotation, {"type": "string"}))


def _validate_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ToolArgumentError(
            f"Tool arguments for {tool.name} must be an object."
        )
    try:
        _validate_schema(arguments, tool.input_schema, path="")
    except ToolArgumentError as exc:
        raise ToolArgumentError(
            f"Tool arguments for {tool.name} are invalid: {exc}"
        ) from exc


def _validate_schema(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if "anyOf" in schema:
        errors: list[str] = []
        for option in schema["anyOf"]:
            try:
                _validate_schema(value, option, path=path)
                return
            except ToolArgumentError as exc:
                errors.append(str(exc))
        expected_types = [
            option.get("type")
            for option in schema["anyOf"]
            if isinstance(option, dict) and isinstance(option.get("type"), str)
        ]
        if expected_types:
            raise ToolArgumentError(
                f"{_label(path)} must be {_type_label(expected_types)}; "
                f"got {type(value).__name__}."
            )
        raise ToolArgumentError(" or ".join(errors))

    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentError(
            f"{_label(path)} must be one of {schema['enum']!r}; got {value!r}."
        )

    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise ToolArgumentError(
            f"{_label(path)} must be {_type_label(expected_type)}; "
            f"got {type(value).__name__}."
        )

    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ToolArgumentError(f"missing required field: {_join(path, key)}.")
        if schema.get("additionalProperties") is False:
            allowed = set(properties)
            extra = sorted(set(value) - allowed)
            if extra:
                raise ToolArgumentError(
                    f"unexpected field: {_join(path, extra[0])}."
                )
        for key, nested_value in value.items():
            nested_schema = properties.get(key)
            if isinstance(nested_schema, dict):
                _validate_schema(nested_value, nested_schema, path=_join(path, key))

    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, path=f"{path}[{index}]")


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    if expected_type == "null":
        return value is None
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def _type_label(expected_type: Any) -> str:
    if isinstance(expected_type, list):
        return " or ".join(str(item) for item in expected_type)
    return str(expected_type)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _label(path: str) -> str:
    return path or "value"
