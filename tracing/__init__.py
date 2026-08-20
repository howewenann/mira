"""Optional generic OTLP tracing support."""

from tracing.bootstrap import configure_tracing, parse_headers_yaml, tracing_yaml_fragment

__all__ = ["configure_tracing", "parse_headers_yaml", "tracing_yaml_fragment"]
