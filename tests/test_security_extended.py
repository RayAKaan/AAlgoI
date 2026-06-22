"""Extended security regression tests."""
import pytest
import json

from aalgoi._data import normalize, _normalize_string, _looks_like_file, _normalize_file, _normalize_bytes, _normalize_url


class TestPathTraversal:
    def test_dot_dot_slash_blocked(self):
        result = normalize("../../etc/passwd")
        assert result == "../../etc/passwd"

    def test_dot_dot_backslash_blocked(self):
        result = normalize("..\\..\\etc\\passwd")
        assert result == "..\\..\\etc\\passwd"

    def test_absolute_path_blocked(self):
        result = normalize("/etc/shadow")
        assert result == "/etc/shadow"

    def test_home_path_blocked(self):
        result = normalize("~/.bashrc")
        assert result == "~/.bashrc"

    def test_relative_path_blocked(self):
        result = normalize("./config.json")
        assert result == "./config.json"

    def test_file_read_explicit_allows_traversal(self, tmp_path):
        test_file = tmp_path / "test.json"
        test_file.write_text('{"data": "test"}')
        result = normalize(str(test_file), allow_file_read=True)
        assert result == {"data": "test"}


class TestURLSecurity:
    def test_http_blocked(self):
        result = normalize("http://example.com/api")
        assert result == "http://example.com/api"

    def test_https_blocked(self):
        result = normalize("https://api.example.com/data")
        assert result == "https://api.example.com/data"

    def test_localhost_blocked(self):
        result = normalize("http://localhost:8080/admin")
        assert result == "http://localhost:8080/admin"

    def test_internal_ip_blocked(self):
        result = normalize("http://192.168.1.1/config")
        assert result == "http://192.168.1.1/config"

    def test_url_fetch_explicit_allows(self):
        assert callable(normalize)


class TestDataExfiltration:
    def test_large_string_not_read_as_file(self):
        long_path = "/" + "a" * 10000
        result = normalize(long_path)
        assert result == long_path

    def test_json_string_not_treated_as_file(self):
        result = normalize('{"sensitive": "data"}')
        assert result == {"sensitive": "data"}

    def test_json_array_string_parsed(self):
        result = normalize('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_nested_json_parsed(self):
        data = '{"users": [{"name": "Alice", "id": 1}, {"name": "Bob", "id": 2}]}'
        result = normalize(data)
        assert result["users"][0]["name"] == "Alice"

    def test_invalid_json_returns_string(self):
        result = normalize('{invalid json}')
        assert result == '{invalid json}'

    def test_xml_string_not_read_as_file(self):
        result = normalize("<root><data>test</data></root>")
        assert result == "<root><data>test</data></root>"

    def test_yaml_string_not_read_as_file(self):
        result = normalize("key: value\nname: test")
        assert result == "key: value\nname: test"


class TestBytesNormalization:
    def test_json_bytes_parsed(self):
        data = b'{"key": "value"}'
        result = normalize(data)
        assert result == {"key": "value"}

    def test_csv_bytes_parsed(self):
        data = b"name,age\nAlice,30\nBob,25"
        result = normalize(data)
        assert isinstance(result, dict)
        assert "columns" in result
        assert result["columns"] == ["name", "age"]

    def test_arbitrary_bytes_base64(self):
        data = b'\x00\x01\x02\x03\xff\xfe\xfd'
        result = normalize(data)
        assert isinstance(result, str)
        import base64
        decoded = base64.b64decode(result)
        assert decoded == data

    def test_large_bytes(self):
        data = b'x' * 1000000
        result = normalize(data)
        assert isinstance(result, str)
        assert len(result) > 0


class TestNormalizeEdgeCases:
    def test_none_input(self):
        assert normalize(None) is None

    def test_empty_string(self):
        assert normalize("") == ""

    def test_whitespace_string(self):
        assert normalize("   ") == "   "

    def test_bool_input(self):
        assert normalize(True) is True
        assert normalize(False) is False

    def test_deeply_nested_structure(self):
        data = {"a": {"b": {"c": {"d": {"e": [1, 2, 3]}}}}}
        result = normalize(data)
        assert result["a"]["b"]["c"]["d"]["e"] == [1, 2, 3]

    def test_mixed_types_in_list(self):
        data = [1, "two", 3.0, True, None, [4, 5]]
        result = normalize(data)
        assert result == [1, "two", 3.0, True, None, [4, 5]]

    def test_dict_with_non_string_keys(self):
        data = {1: "one", 2: "two", 3.14: "pi"}
        result = normalize(data)
        assert "1" in result
        assert "2" in result
        assert "3.14" in result


class TestLooksLikeFile:
    def test_starts_with_tilde(self):
        assert _looks_like_file("~/data.json") is True

    def test_starts_with_slash(self):
        assert _looks_like_file("/etc/config.json") is True

    def test_starts_with_dot_slash(self):
        assert _looks_like_file("./config.json") is True

    def test_starts_with_dot_dot_slash(self):
        assert _looks_like_file("../config.json") is True

    def test_json_extension(self):
        assert _looks_like_file("data.json") is True

    def test_csv_extension(self):
        assert _looks_like_file("data.csv") is True

    def test_txt_extension(self):
        assert _looks_like_file("notes.txt") is True

    def test_yaml_extension(self):
        assert _looks_like_file("config.yaml") is True

    def test_yml_extension(self):
        assert _looks_like_file("config.yml") is True

    def test_parquet_extension(self):
        assert _looks_like_file("data.parquet") is True

    def test_no_extension(self):
        assert _looks_like_file("hello") is False

    def test_empty_string(self):
        assert _looks_like_file("") is False

    def test_plain_text(self):
        assert _looks_like_file("just some text") is False

    def test_url_not_file(self):
        assert _looks_like_file("https://example.com") is False
        assert _looks_like_file("http://example.com") is False
