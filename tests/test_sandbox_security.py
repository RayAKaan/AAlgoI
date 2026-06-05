import pytest

from aalgoi.core.sandboxed_executor import (
    SandboxValidator,
    create_sandboxed_module,
    execute_sandboxed,
    DANGEROUS_MODULES,
)


def test_blocks_os_import():
    code = "import os; def process(data): return os.system('echo pwned')"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_subprocess_import():
    code = "import subprocess; def process(data): return subprocess.run(['ls'])"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_eval():
    code = "def process(data): return eval(data)"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_exec():
    code = """def process(data):
    exec("import os")
    return data"""
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_open():
    code = "def process(data): return open('/etc/passwd')"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_socket_import():
    code = "import socket; def process(data): return socket.gethostname()"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_compile():
    code = "def process(data): return compile('1+1', '<string>', 'eval')"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_threading():
    code = "import threading; def process(data): return threading.active_count()"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_allows_safe_builtins():
    code = """
def process(data):
    return sorted([x * 2 for x in data])
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", [1, 2, 3])
    assert success
    assert result == [2, 4, 6]


def test_allows_arithmetic():
    code = """
def process(data):
    return sum(data) / len(data) if data else 0
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", [1, 2, 3])
    assert success
    assert result == 2.0


def test_allows_type_checking():
    code = """
def process(data):
    return isinstance(data, list) and len(data)
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", [1, 2, 3])
    assert success
    assert result == 3


def test_timeout_enforced():
    code = """
def process(data):
    while True:
        _ = 1
    return data
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(
        module, "process", [1, 2, 3], timeout=0.5
    )
    assert not success


def test_infinite_loop_timeout():
    code = """
def process(data):
    x = 0
    while True:
        x += 1
    return x
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(
        module, "process", [1], timeout=0.5
    )
    assert not success


def test_blocks_shutil_import():
    code = "import shutil; def process(data): return shutil.rmtree('/')"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_pickle_import():
    code = "import pickle; def process(data): return pickle.loads(data)"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_ctypes_import():
    code = "import ctypes; def process(data): return None"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_pathlib_import():
    code = "import pathlib; def process(data): return None"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_dunder_builtins():
    code = """
def process(data):
    a = __builtins__
    return data
"""
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_dunder_import_call():
    code = """
def process(data):
    return __import__('os')
"""
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_urllib_import():
    code = "import urllib.request; def process(data): return None"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_execute_sandboxed_no_function_returns_false():
    code = """
def process(data):
    return data
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "nonexistent", [1])
    assert not success
    assert result is None


def test_execute_sandboxed_runtime_error():
    code = """
def process(data):
    return data + 1
"""
    module = create_sandboxed_module("test", code)
    assert module is not None
    success, result = execute_sandboxed(module, "process", "hello")
    assert not success


def test_blocks_importlib():
    code = "import importlib; def process(data): return None"
    module = create_sandboxed_module("test", code)
    assert module is None


def test_blocks_sys_import():
    code = "import sys; def process(data): return sys.path"
    module = create_sandboxed_module("test", code)
    assert module is None


@pytest.mark.parametrize("mod", sorted(DANGEROUS_MODULES))
def test_blocks_all_dangerous_modules(mod):
    code = f"import {mod}; def process(data): return None"
    module = create_sandboxed_module("test", code)
    assert module is None, f"Module {mod} was not blocked"
