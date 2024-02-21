import pytest
from utils import add, subtract, multiply, divide, exception_handler

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(-1, -1) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6

def test_divide():
    assert divide(10, 2) == 5
    assert divide(5, 2) == 2.5
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

def test_exception_handler():
    @exception_handler
    def error_function():
        raise ValueError("Error intencional")

    with pytest.raises(ValueError) as exc_info:
        error_function()
    assert "Error intencional" in str(exc_info.value)
