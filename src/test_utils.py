import pytest
from src.utils import exception_handler

def test_exception_handler():
    @exception_handler
    def error_function():
        raise ValueError("Error intencional")

    with pytest.raises(ValueError) as exc_info:
        error_function()
    assert "Error intencional" in str(exc_info.value)
