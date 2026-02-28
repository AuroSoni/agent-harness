"""Schema generation utilities for tool functions.

Analyzes Python function type hints and Google-style docstrings to produce
canonical tool schemas in the format: {"name", "description", "input_schema"}.
"""
import inspect
import types
import re
from copy import copy
from typing import get_type_hints, get_origin, get_args, Union, Literal, Any, Callable


_BASE_TYPE_MAPPING = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    Any: {},  # Empty schema = unconstrained/any value (JSON Schema 2020-12 compliant)
    types.NoneType: {"type": "null"},
}


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas."""


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas."""


def _parse_union_type(args: tuple[Any, ...]) -> dict:
    subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
    if len(subtypes) == 1:
        return_dict = subtypes[0]
    elif all(isinstance(subtype.get("type"), str) for subtype in subtypes):
        return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
    else:
        return_dict = {"anyOf": subtypes}

    # Handle nullable types by adding "null" to the type (JSON Schema 2020-12 compliant)
    if type(None) in args:
        if "type" in return_dict and isinstance(return_dict["type"], list):
            if "null" not in return_dict["type"]:
                return_dict["type"].append("null")
        elif "type" in return_dict:
            return_dict["type"] = [return_dict["type"], "null"]
        elif "anyOf" in return_dict:
            return_dict["anyOf"].append({"type": "null"})
    return return_dict


def _get_json_schema_type(param_type: type) -> dict[str, str]:
    if param_type in _BASE_TYPE_MAPPING:
        return copy(_BASE_TYPE_MAPPING[param_type])
    return {"type": "object"}


def _parse_type_hint(hint: type) -> dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException(
                "Couldn't parse this type hint, likely due to a custom class or object: ",
                hint,
            )

    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        return _parse_union_type(args)

    elif origin is list:
        if not args:
            return {"type": "array"}
        else:
            return {"type": "array", "items": _parse_type_hint(args[0])}

    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace('typing.', '')} is a Tuple with a single element, which "
                "we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain "
                "more than one element, we recommend "
                "using a List[] type instead, or if it really is a single element, remove the Tuple[] wrapper and just "
                "pass the element directly."
            )
        if ... in args:
            raise TypeHintParsingException(
                "Conversion of '...' is not supported in Tuple type hints. "
                "Use List[] types for variable-length inputs instead."
            )
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}

    elif origin is dict:
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out

    elif origin is Literal:
        literal_types = set(type(arg) for arg in args)
        final_type = _parse_union_type(tuple(literal_types))
        final_type.update({"enum": [arg for arg in args if arg is not None]})
        return final_type

    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object: ", hint)


# Regex patterns for parsing Google-style docstrings.

# Extracts the initial segment of the docstring, containing the function description
description_re = re.compile(r"^(.*?)(?=\n\s*(Args:|Returns:|Raises:)|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"(?:^|\n)"
    r"\s*(\w+)\s*(?:\([^)]*?\))?:\s*"
    r"(.*?)\s*"
    r"(?=\n\s*\w+\s*(?:\([^)]*?\))?:|\Z)",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present
returns_re = re.compile(
    r"\n\s*Returns:\n\s*"
    r"(?:[^)]*?:\s*)?"
    r"(.*?)"
    r"[\n\s]*(Raises:|\Z)",
    re.DOTALL,
)


def _parse_google_format_docstring(
    docstring: str,
) -> tuple[str | None, dict | None, str | None]:
    """Parses a Google-style docstring to extract the function description,
    argument descriptions, and return description.

    Args:
        docstring: The docstring to parse.

    Returns:
        Tuple of (description, args_dict, returns_description).
    """
    description_match = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)

    description = description_match.group(1).strip() if description_match else None
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0]: re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
    else:
        args_dict = {}

    return description, args_dict, returns


def _convert_type_hints_to_json_schema(func: callable, error_on_missing_type_hints: bool = True) -> dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty and error_on_missing_type_hints:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param_name not in properties:
            properties[param_name] = {}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # Return: multi-type union -> treat as unconstrained (empty schema)
    if (
        "return" in properties
        and (return_type := properties["return"].get("type"))
        and not isinstance(return_type, str)
    ):
        properties["return"] = {}

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def generate_tool_schema(func: Callable) -> dict:
    """Generate a canonical tool schema from a function's type hints and docstring.

    The schema format is: {"name", "description", "input_schema"}.
    This is the framework's internal canonical form — provider-specific
    conversions (Anthropic, OpenAI) happen at schema export time.

    Args:
        func: The function to generate a schema for. Must have type hints
            for all parameters. Google-style docstring is preferred but
            a fallback description is generated if missing.

    Returns:
        Schema dict with "name", "description", and "input_schema" keys.

    Raises:
        TypeHintParsingException: If type hints are missing or cannot be parsed.
    """
    func_name = func.__name__
    doc = inspect.getdoc(func)

    description = None
    param_descriptions: dict[str, str] = {}
    if doc:
        try:
            description, param_descriptions, _ = _parse_google_format_docstring(doc)
        except Exception:
            description = doc.split("\n")[0].strip()
            param_descriptions = {}

    if not description:
        description = f"Function: {func_name}"

    try:
        json_schema = _convert_type_hints_to_json_schema(func, error_on_missing_type_hints=True)
    except TypeHintParsingException as e:
        raise TypeHintParsingException(
            f"Failed to generate schema for {func_name}: {str(e)}"
        ) from e

    json_schema["properties"].pop("return", None)

    for param_name, param_schema in json_schema["properties"].items():
        if param_name in param_descriptions:
            param_schema["description"] = param_descriptions[param_name]
        else:
            param_type = param_schema.get("type", "any")
            param_schema["description"] = f"Parameter of type {param_type}"

    return {
        "name": func_name,
        "description": description,
        "input_schema": json_schema,
    }
