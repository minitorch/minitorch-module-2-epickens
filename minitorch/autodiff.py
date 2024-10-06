from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    # Step 1: Compute f(x_0, ..., x_{i-1}, x_i + epsilon / 2, x_{i+1}, ..., x_{n-1})
    forward_diff = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])

    # Step 2: Compute f(x_0, ..., x_{i-1}, x_i - epsilon / 2, x_{i+1}, ..., x_{n-1})
    backward_diff = f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])

    # Step 3: Compute the central difference
    return (forward_diff - backward_diff) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    result = []
    visited = set()

    def visit(v: Variable) -> None:
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            visit(parent)
        result.append(v)

    visit(variable)

    if len(result) != len(visited):
        raise ValueError("Cycle detected in the computation graph.")

    result.reverse()

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # Step 1: Get the topological order of the computation graph
    variables = topological_sort(variable)

    # Step 2: Create a dictionary to store the derivative of each variable
    derivative_map = {variable.unique_id: deriv}

    # Step 3: Propagate the derivative backward
    for v in variables:
        if v.parents:
            for parent, derivative in v.chain_rule(derivative_map[v.unique_id]):
                if parent.unique_id in derivative_map:
                    derivative_map[parent.unique_id] += derivative
                else:
                    derivative_map[parent.unique_id] = derivative

    # Step 4: Accumulate the derivative to the leaf nodes
    for v in variables:
        if v.is_leaf():
            v.accumulate_derivative(derivative_map[v.unique_id])


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
