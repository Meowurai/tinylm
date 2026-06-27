# tests/test_autograd.py

import math

from tinylm.autograd import Value

def test_value_stores_data_and_grad():
    x = Value(2.0)

    assert x.data == 2.0
    assert x.grad == 0.0

def test_value_repr():
    x = Value(2.0)

    assert repr(x) == "Value(data=2.0, grad=0.0)"


def test_add_forward():
    x = Value(2.0)
    y = Value(3.0)

    z = x + y 

    assert z.data == 5.0

def test_add_backward():
    x = Value(2.0)
    y = Value(3.0)

    z = x + y
    z.grad = 1.0
    z._backward()

    assert x.grad == 1.0
    assert y.grad == 1.0


def test_mul_forward():
    x = Value(2.0)
    y = Value(3.0)

    z = x * y

    assert z.data == 6.0

def test_mul_backward():
    x = Value(2.0)
    y = Value(3.0)

    z = x * y
    z.grad = 1.0
    z._backward()

    assert x.grad == 3.0
    assert y.grad == 2.0

def test_exp_forward():
    x = Value(2.0)

    y = x.exp()

    assert y.data == math.exp(2.0)

def test_exp_backward():
    x = Value(2.0)

    y = x.exp()
    y.grad = 1.0
    y._backward()

    assert y.grad == 1.0
    assert x.grad == math.exp(2.0)

def test_log_backward():
    x = Value(2.0)

    y = x.log()
    y.grad = 1.0
    y._backward()

    assert x.grad == 1 / 2.0