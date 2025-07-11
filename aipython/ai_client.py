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
        self.client = Client(api_key=self.api_key)
    
    def generate_code(self, description: str, model: Optional[str] = None) -> str:
        """Generate Python code from natural language description.
        
        Args:
            description: What the user wants to accomplish
            model: Override default model
            
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
- Don't include print statements unless specifically requested"""

        try:
            response = self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate Python code to: {description}"}
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
    
    def ask_question(self, question: str, model: Optional[str] = None) -> str:
        """Ask the AI a question.
        
        Args:
            question: User's question
            model: Override default model
            
        Returns:
            AI's response
        """
        model = model or self.model
        
        system_prompt = """You are a helpful Python/data science assistant. 
Answer questions clearly and concisely. If the question is about code or data analysis, 
provide practical advice and examples when helpful."""

        try:
            response = self.client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ], model=model)
            
            return response.content[0].text.strip()
            
        except Exception as e:
            raise Exception(f"Failed to get answer: {e}")
    
    def get_ipython_context(self) -> dict:
        """Get current IPython context (variables, history, etc.).
        
        This will be expanded in Phase 2 to provide context to the AI.
        """
        from IPython import get_ipython
        
        ip = get_ipython()
        if not ip:
            return {}
            
        # Get user namespace variables
        user_vars = {k: type(v).__name__ for k, v in ip.user_ns.items() 
                    if not k.startswith('_')}
        
        return {
            'variables': user_vars,
            'history_length': len(ip.history_manager.get_range())
        }
