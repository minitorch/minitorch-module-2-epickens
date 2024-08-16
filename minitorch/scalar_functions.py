from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Forward multiplication $f(x, y) = x \times y$

        Args:
            ctx: autodiff context
            a: float
            b: float

        Returns:
            float: $a \times b$
        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Compute the gradient of the multiplication function.
        f'_a (a,b) = b, f'_b (a,b) = a

        Args:
            ctx: autodiff context
            d_output: gradient of the output
        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Forward inverse $f(x) = 1/x$

        Args:
            ctx: autodiff context
            a: float

        Returns:
            float: $1/x$
        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Compute the gradient of the inverse function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            float: $d \times f'(x) = -1/x^2$
        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Forward negation $f(x) = -x$

        Args:
            ctx: autodiff context
            a: float

        Returns:
            float: $-x$
        """
        return -1.0 * a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Compute the gradient of the negation function. Note: $f'(x) = -1$

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            float: $d \times f'(x) = -1$
        """
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute the forward pass of the sigmoid function.

        Args:
            ctx: autodiff context
            a: float

        Returns:
            float: $1 / (1 + exp(-a))$ or $exp(a) / (1 + exp(a))$ depending on the value of $a$
        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Compute the gradient of the sigmoid function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            float: $d \times f'(x)$ where $f'(x) = exp(a) / (1 + exp(a))^2$
        """
        (a,) = ctx.saved_values
        if a < 0:
            back_a = operators.exp(a) / (1 + operators.exp(a)) ** 2
        else:
            back_a = operators.exp(-a) / (1 + operators.exp(-a)) ** 2
        return d_output * back_a


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute the forward pass of the ReLU function.

        Args:
            ctx: autodiff context
            a: float

        Returns:
            float: $max(0, a)$
        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Compute the gradient of the ReLU function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            float: $d \times f'(x)$ where $f'(x) = 1$ if $a > 0$ else $0$
        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute the forward pass of the exponential function.

        Args:
            ctx: autodiff context
            a: float

        Returns:
            float: $exp(a)$
        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Compute the gradient of the exponential function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            float: $d \times f'(x)$ where $f'(x) = exp(x)$
        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Compute the forward pass of the less-than function.

        Args:
            ctx: autodiff context
            a: float
            b: float

        Returns:
            float: 1.0 if a is less than b else 0.0
        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Compute the gradient of the less-than function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            Tuple[float, flaot]: 0.0, 0.0
        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Compute the forward pass of the equal function.

        Args:
            ctx: autodiff context
            a: float
            b: float

        Returns:
            float: 1.0 if a is equal to b else 0.0
        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Compute the gradient of the equal function.

        Args:
            ctx: autodiff context
            d_output: gradient of the output

        Returns:
            Tuple[float, flaot]: 0.0, 0.0
        """
        return 0.0, 0.0
