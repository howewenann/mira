"""Schema-driven form controls for native MCP elicitation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Input, Select, SelectionList, Static

FieldKind = Literal["string", "integer", "number", "boolean", "enum", "multi_enum"]


@dataclass(frozen=True, slots=True)
class MCPFormField:
    """One supported top-level property from an MCP requested schema."""

    key: str
    title: str
    description: str
    kind: FieldKind
    required: bool
    choices: tuple[Any, ...] = ()
    default: Any = None


class MCPFormValidationError(ValueError):
    """Validation failure tied to one rendered form field."""

    def __init__(self, field_id: str, message: str) -> None:
        super().__init__(message)
        self.field_id = field_id


def mcp_form_fields(schema: dict[str, Any]) -> tuple[MCPFormField, ...]:
    """Project supported top-level JSON Schema properties into form fields."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ()
    raw_required = schema.get("required")
    required = (
        {key for key in raw_required if isinstance(key, str)}
        if isinstance(raw_required, list)
        else set()
    )
    fields = []
    for key, raw_property in properties.items():
        if not isinstance(key, str) or not isinstance(raw_property, dict):
            continue
        kind, choices = _field_kind(raw_property)
        fields.append(
            MCPFormField(
                key=key,
                title=str(raw_property.get("title") or _field_title(key)),
                description=str(raw_property.get("description") or ""),
                kind=kind,
                required=key in required,
                choices=choices,
                default=raw_property.get("default"),
            )
        )
    return tuple(fields)


def _field_kind(schema: dict[str, Any]) -> tuple[FieldKind, tuple[Any, ...]]:
    field_type = schema.get("type")
    if isinstance(field_type, list):
        field_type = next((value for value in field_type if value != "null"), "string")
    if field_type == "array":
        items = schema.get("items")
        choices = _enum_choices(items.get("enum")) if isinstance(items, dict) else ()
        return ("multi_enum", choices) if choices else ("string", ())
    choices = _enum_choices(schema.get("enum"))
    if choices:
        return "enum", choices
    if field_type in {"integer", "number", "boolean"}:
        return field_type, ()
    return "string", ()


def _enum_choices(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item
        for item in value
        if isinstance(item, (str, int, float, bool)) and not isinstance(item, (list, dict))
    )


def _field_title(key: str) -> str:
    return key.replace("_", " ").replace("-", " ").strip().capitalize() or "Value"


class MCPElicitationForm(Vertical):
    """Render and collect one MCP form-mode elicitation schema."""

    def __init__(self, schema: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fields = mcp_form_fields(schema)

    def compose(self) -> ComposeResult:
        if not self.fields:
            yield Static("No values are requested.", classes="mcp-form-empty")
        for index, field in enumerate(self.fields):
            with Vertical(classes="mcp-form-field"):
                marker = " *" if field.required else ""
                yield Static(
                    f"{field.title}{marker}",
                    classes="mcp-form-label",
                    markup=False,
                )
                if field.description:
                    yield Static(
                        field.description,
                        classes="mcp-form-description",
                        markup=False,
                    )
                yield self._field_widget(index, field)
        yield Static("", id="mcp-form-error", classes="mcp-form-error", markup=False)

    def _field_widget(self, index: int, field: MCPFormField) -> Any:
        widget_id = self.field_id(index)
        if field.kind == "boolean":
            return Checkbox(
                "Yes",
                value=field.default if isinstance(field.default, bool) else False,
                id=widget_id,
                classes="mcp-form-checkbox",
            )
        if field.kind == "enum":
            value = field.default if field.default in field.choices else Select.NULL
            return Select(
                [(str(choice), choice) for choice in field.choices],
                prompt="Choose a value",
                allow_blank=True,
                value=value,
                id=widget_id,
                classes="mcp-form-select",
            )
        if field.kind == "multi_enum":
            defaults = field.default if isinstance(field.default, list) else []
            return SelectionList(
                *[
                    (str(choice), choice, choice in defaults)
                    for choice in field.choices
                ],
                id=widget_id,
                classes="mcp-form-selection-list",
            )
        default = "" if field.default is None else str(field.default)
        return Input(
            value=default,
            type="number" if field.kind in {"integer", "number"} else "text",
            id=widget_id,
            classes="mcp-form-input",
        )

    def collect_values(self) -> dict[str, Any]:
        """Return typed content or raise the first actionable validation error."""
        self.query_one("#mcp-form-error", Static).update("")
        values: dict[str, Any] = {}
        for index, field in enumerate(self.fields):
            widget_id = self.field_id(index)
            widget = self.query_one(f"#{widget_id}")
            value = self._field_value(field, widget, widget_id)
            if value is not None:
                values[field.key] = value
        return values

    def _field_value(self, field: MCPFormField, widget: Any, widget_id: str) -> Any:
        if field.kind == "boolean":
            return bool(widget.value)
        if field.kind == "enum":
            if widget.value is Select.NULL:
                if field.required:
                    raise MCPFormValidationError(widget_id, f"{field.title} is required.")
                return None
            return widget.value
        if field.kind == "multi_enum":
            selected = list(widget.selected)
            if field.required and not selected:
                raise MCPFormValidationError(widget_id, f"{field.title} is required.")
            return selected

        raw = str(widget.value)
        if not raw.strip():
            if field.required:
                raise MCPFormValidationError(widget_id, f"{field.title} is required.")
            return None
        if field.kind == "integer":
            try:
                return int(raw)
            except ValueError as error:
                raise MCPFormValidationError(
                    widget_id,
                    f"{field.title} must be an integer.",
                ) from error
        if field.kind == "number":
            try:
                value = float(raw)
            except ValueError as error:
                raise MCPFormValidationError(
                    widget_id,
                    f"{field.title} must be a number.",
                ) from error
            if not math.isfinite(value):
                raise MCPFormValidationError(widget_id, f"{field.title} must be a finite number.")
            return value
        return raw

    def show_validation_error(self, error: MCPFormValidationError) -> None:
        """Display one validation error and focus its field."""
        self.query_one("#mcp-form-error", Static).update(str(error))
        self.query_one(f"#{error.field_id}").focus()

    def focus_first_field(self) -> None:
        """Focus the first editable schema field."""
        if self.fields:
            self.query_one(f"#{self.field_id(0)}").focus()

    @staticmethod
    def field_id(index: int) -> str:
        return f"mcp-form-field-{index}"


__all__ = [
    "MCPElicitationForm",
    "MCPFormField",
    "MCPFormValidationError",
    "mcp_form_fields",
]
