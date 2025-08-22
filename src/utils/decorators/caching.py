import streamlit as st
from functools import wraps

def cached_resource(show_spinner=False):
    """Streamlit resource cache wrapper decorator."""
    def decorator(fn):
        return st.cache_resource(show_spinner=show_spinner)(fn)
    return decorator

def cached_data(show_spinner=False):
    """Streamlit data cache wrapper decorator."""
    def decorator(fn):
        return st.cache_data(show_spinner=show_spinner)(fn)
    return decorator
