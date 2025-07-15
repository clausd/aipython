"""AI client wrapper using Claudette."""

import os
from typing import Optional
from claudette import Client


class AIClient:
    """Wrapper for AI interactions using Claudette."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'claude-3-sonnet-20240229'):
        """Initialize AI client.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.client = Client(self.api_key)
    
    def generate_code(self, description: str, model: Optional[str] = None, context: Optional[dict] = None) -> str:
        """Generate Python code from natural language description.
        
        Args:
            description: What the user wants to accomplish
            model: Override default model
            context: Current IPython context (variables, etc.)
            
        Returns:
            Generated Python code as string
        """
        model = model or self.model
        
        system_prompt = """You are a Python code generator for IPython/Jupyter. 
Generate clean, executable Python code based on the user's description.

Rules:
- Return ONLY the Python code, no explanations or markdown
- Code should be ready to execute in IPython
- Use common libraries (pandas, numpy, matplotlib, etc.) when appropriate
- Keep code concise but readable
- Don't include print statements unless specifically requested
- Use existing variables when appropriate based on the provided context"""

        # Build user message with context
        user_message = f"Generate Python code to: {description}"
        
        if context and context.get('variables'):
            user_message += f"\n\nAvailable variables in current session:\n"
            for var_name, var_info in context['variables'].items():
                user_message += f"- {var_name}: {var_info['type']}"
                if var_info.get('shape'):
                    user_message += f" {var_info['shape']}"
                if var_info.get('preview'):
                    user_message += f" = {var_info['preview']}"
                user_message += "\n"

        try:
            response = self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ], model=model)
            
            # Extract code from response
            code = response.content[0].text.strip()
            
            # Clean up any markdown code blocks
            if code.startswith('```python'):
                code = code[9:]  # Remove ```python
            if code.startswith('```'):
                code = code[3:]   # Remove ```
            if code.endswith('```'):
                code = code[:-3]  # Remove trailing ```
                
            return code.strip()
            
        except Exception as e:
            raise Exception(f"Failed to generate code: {e}")
    
    def ask_question(self, question: str, model: Optional[str] = None, context: Optional[dict] = None) -> str:
        """Ask the AI a question.
        
        Args:
            question: User's question
            model: Override default model
            context: Current IPython context (variables, etc.)
            
        Returns:
            AI's response
        """
        model = model or self.model
        
        system_prompt = """You are a helpful Python/data science assistant. 
Answer questions clearly and concisely. If the question is about code or data analysis, 
provide practical advice and examples when helpful. Use the provided context about 
the user's current session to give more relevant answers."""

        # Build user message with context
        user_message = question
        
        if context and context.get('variables'):
            user_message += f"\n\nCurrent session context:\n"
            for var_name, var_info in context['variables'].items():
                user_message += f"- {var_name}: {var_info['type']}"
                if var_info.get('shape'):
                    user_message += f" {var_info['shape']}"
                if var_info.get('preview'):
                    user_message += f" = {var_info['preview']}"
                user_message += "\n"

        try:
            response = self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ], model=model)
            
            return response.content[0].text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to get answer: {e}")
    
    def get_ipython_context(self) -> dict:
        """Get current IPython context (variables, history, etc.)."""
        from IPython import get_ipython
        import pandas as pd
        import numpy as np
        
        ip = get_ipython()
        if not ip:
            return {}
            
        # Get user namespace variables with enhanced info
        variables = {}
        for k, v in ip.user_ns.items():
            if k.startswith('_') or k in ['In', 'Out', 'get_ipython', 'exit', 'quit']:
                continue
                
            var_info = {'type': type(v).__name__}
            
            # Add shape info for arrays/dataframes
            if hasattr(v, 'shape'):
                var_info['shape'] = str(v.shape)
            elif hasattr(v, '__len__') and not isinstance(v, str):
                try:
                    var_info['shape'] = f"(len={len(v)})"
                except:
                    pass
            
            # Add preview for small values
            try:
                if isinstance(v, pd.DataFrame):
                    var_info['preview'] = f"columns: {list(v.columns)}"
                elif isinstance(v, pd.Series):
                    var_info['preview'] = f"name: {v.name}"
                elif isinstance(v, np.ndarray):
                    var_info['preview'] = f"dtype: {v.dtype}"
                else:
                    str_repr = str(v)
                    if len(str_repr) <= 100:
                        var_info['preview'] = str_repr
            except:
                pass
                
            variables[k] = var_info
        
        return {
            'variables': variables,
            'history_length': len(ip.history_manager.get_range()) if ip.history_manager else 0
        }
