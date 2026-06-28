# src/tinylm/autograd.py

# The job of autograd.py is to provide a scalar Value object
# that tracks computations and supports reverse-mode autodiff

from __future__ import annotations

import math

class Value:
    def __init__(
        self, data: float, 
        _children=(),
        _op=""
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other: Value) -> Value:
        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out
    
    def __mul__(self, other: Value) -> Value:
        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __truediv__(self, other: Value) -> Value:
        out = Value(
            self.data / other.data,
            (self, other),
            "/"
        )

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad += -self.data / (other.data ** 2) * out.grad

        out._backward = _backward

        return out
    
    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out
    
    def log(self) -> Value: 
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out
    
    def tanh(self) -> Value:
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out
    
    def backward(self):
        """
        Backward start from this Value and propagates gradients
        backward through every Value that contributed to it.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for parent in v._prev:
                    build_topo(parent)

                topo.append(v)
        
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
