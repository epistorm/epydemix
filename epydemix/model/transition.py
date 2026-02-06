from typing import Any


class Transition:
    """
    Represents a transition in an epidemic model.

    A transition defines a change from one compartment to another with a specified rate. Optionally, it can involve a specific agent.

    Attributes:
        source (str): The source compartment of the transition.
        target (str): The target compartment of the transition.
        kind (str): The kind of transition (e.g., spontaneous or mediated).
        params (Any): The parameters involved in the transition.
    """

    def __init__(self, source: str, target: str, kind: str, params: Any) -> None:
        """
        Initializes the Transition object.

        Args:
            source (str): The source compartment of the transition.
            target (str): The target compartment of the transition.
            kind (str): The kind of transition (e.g., spontaneous or mediated).
            params (Any): The parameters involved in the transition.

        Returns:
            None
        """
        self.source = source
        self.target = target
        self.kind = kind
        self.params = params
