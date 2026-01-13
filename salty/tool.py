"""Tool definition API for Salty."""

from typing import Callable, Dict, Any


class Tool:
    """A tool that the LLM can call during conversation."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict[str, Dict[str, str]] | None = None,
        required: list[str] | None = None,
    ):
        """Create a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            function: Python function to call
            parameters: Dict of parameter_name -> {type, description}
            required: List of required parameter names
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters or {}
        self.required = required or []

    def build(self) -> Dict[str, Any]:
        """Build OpenAI tool specification."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }
