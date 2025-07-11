"""AI-enhanced IPython with natural language code generation."""

__version__ = "0.1.0"

def load_ipython_extension(ipython):
    """Load the aipython extension in IPython."""
    from .magic import AIMagics
    ipython.register_magic_function(AIMagics(ipython).ai, 'line', 'ai')
    ipython.register_magic_function(AIMagics(ipython).ai_question, 'line', 'ai?')
    print("🤖 AI Python extension loaded! Try: %ai create a list of numbers")
