# AI Python - AI-Enhanced IPython

Transform your IPython experience with AI-powered code generation and assistance.

## Quick Start

1. **Install**:
   ```bash
   pip install aipython
   ```

2. **Set up your Anthropic API key**:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

3. **Launch IPython** - the extension loads automatically:
   ```bash
   ipython
   ```

4. **Try it out**:
   ```python
   %ai create a pandas dataframe with sample sales data
   %ai -e plot a sine wave with matplotlib  
   %ai? what's the difference between a list and a tuple?
   ```

## Magic Commands

- **`%ai <description>`** - Generate Python code from natural language
- **`%ai -e <description>`** - Generate and execute Python code  
- **`%ai? <question>`** - Ask the AI questions about Python/data science

## Examples

```python
# Generate code (shows code, doesn't execute)
%ai create a scatter plot of random data

# Generate and execute immediately
%ai -e load a CSV file called 'data.csv' into a pandas dataframe

# Ask questions
%ai? how do I handle missing values in pandas?
%ai? what does this error mean: KeyError
```

## Requirements

- Python 3.8+
- IPython 7.0+
- Anthropic API key (set as `ANTHROPIC_API_KEY` environment variable)

## Development

```bash
git clone <repo>
cd aipython
pip install -e .
```

## Coming Soon

- Context awareness (access to your variables and history)
- Advanced AI tools and integrations
- Full AI-first REPL mode
- Multi-provider support
