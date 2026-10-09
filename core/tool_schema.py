"""Ollama schema bridge, adapted from Mark-LV PR #112 (Abhayraj69).

Only parameter types are translated; enum values and property names are data.
"""
from copy import deepcopy


def _convert_schema(node):
    if isinstance(node, list):
        return [_convert_schema(value) for value in node]
    if not isinstance(node, dict):
        return node
    out = deepcopy(node)
    if isinstance(out.get("type"), str):
        out["type"] = out["type"].lower()
    elif isinstance(out.get("type"), list):
        out["type"] = [value.lower() for value in out["type"]]
    for key in ("properties", "$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {name: _convert_schema(value) for name, value in out[key].items()}
    for key in ("items", "additionalProperties", "anyOf", "allOf", "oneOf", "not"):
        if key in out:
            out[key] = _convert_schema(out[key])
    return out


def gemini_tools_to_openai(tools):
    return [{"type": "function", "function": {
        "name": tool["name"], "description": tool.get("description", ""),
        "parameters": _convert_schema(tool.get("parameters") or {
            "type": "OBJECT", "properties": {}}),
    }} for tool in tools or [] if isinstance(tool, dict) and tool.get("name")]
