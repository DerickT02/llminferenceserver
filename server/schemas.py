from dataclasses import dataclass, field
from typing import Any

@dataclass
class FieldSchema:
    type: type
    required: bool = True
    default: Any = None
    description: str = ""

@dataclass
class PromptSchema:
    name: str
    template: str
    fields: dict[str, FieldSchema] = field(default_factory=dict)

    def validate(self, inputs: dict) -> dict:
        errors = []

        for field_name, schema in self.fields.items():
            if field_name not in inputs:
                if schema.required:
                    errors.append(f"Missing required field: '{field_name}'")
                else:
                    inputs[field_name] = schema.default
            elif not isinstance(inputs[field_name], schema.type):
                errors.append(
                    f"Field '{field_name}' expected {schema.type.__name__}, "
                    f"got {type(inputs[field_name]).__name__}"
                )

        if errors:
            raise ValueError("\n".join(errors))

        return inputs

    def render(self, inputs: dict) -> str:
        validated = self.validate(inputs)
        return self.template.format(**validated)