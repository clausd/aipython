# AI-Enhanced IPython (aipython)

An AI-powered version of IPython that lets you describe what you want instead of writing code.

## Vision

Transform IPython into an AI-first REPL where you can:
- Describe outcomes instead of writing Python syntax
- Get contextual help with access to your variables and history
- Seamlessly blend AI assistance with normal Python workflow

## Implementation Plan

### Phase 1: Basic AI Magic (MVP) ✅
**Goal**: Prove the concept with simple magic commands

**Features**:
- `%ai <description>` - generates Python code from natural language
- `%ai? <question>` - asks AI questions about code/context  
- `%ai -e <description>` - generates and executes code
- Auto-loads via IPython extension system
- Uses `ANTHROPIC_API_KEY` environment variable

**Installation**:
```bash
pip install aipython
# Auto-loads on IPython startup via entry points
```

**Usage Examples**:
```python
%ai create a pandas dataframe with sample sales data
%ai -e plot a sine wave with matplotlib
%ai? what does the variable 'df' contain?
```

### Phase 2: Enhanced Context & Tools
**Features**:
- Access IPython history and variables as AI tools
- Command output capture and context feeding  
- Error handling and retry logic
- Basic configuration file support

### Phase 3: Full AI-First REPL Mode
**Features**:
- AI-first input mode (your original vision)
- Shell escapes (`!`, `!!`, `?`)
- Configurable instruction limits (3-10 per request)
- History integration with AI instruction tracking

### Phase 4: Advanced Agent Features
**Features**:
- Clipboard integration tools
- File reference and analysis
- Output copying magic (`%copy-output`)
- Multiple AI provider support
- Advanced context management

## Technical Architecture

### Project Structure
```
aipython/
├── __init__.py           # Package init + IPython extension loading
├── magic.py             # AI magic commands
├── ai_client.py         # Claudette wrapper
├── config.py            # Configuration handling
└── tools.py             # Context tools (Phase 2+)
```

### Key Design Decisions

1. **IPython Extension System**: Use standard extension loading via entry points
2. **Environment-based Auth**: Leverage existing `ANTHROPIC_API_KEY` 
3. **Pip Installable**: Standard Python package with proper setup.py
4. **Gradual Enhancement**: Each phase builds on previous functionality
5. **Claudette Integration**: Use for tool definitions and agent loops

### Auto-Loading Strategy

The package will auto-load via IPython's extension system using entry points in setup.py:
```python
entry_points={
    'ipython.extensions': [
        'aipython = aipython:load_ipython_extension',
    ],
}
```

This means after `pip install aipython`, the magic commands are immediately available in any new IPython session.

## Current Focus

**Phase 1 Implementation**: Building the basic `%ai` magic command with:
- Natural language to Python code generation
- Optional execution with `-e` flag
- Question answering with `%ai?`
- Environment-based API key detection
- Proper error handling and user feedback

This gives immediate value while building toward the full AI-first REPL vision.

