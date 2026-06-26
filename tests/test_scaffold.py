"""Scaffold smoke tests."""


def test_package_imports():
    import agent_runtime

    assert agent_runtime.__version__ == "0.1.0"

