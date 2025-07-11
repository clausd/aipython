"""AI magic commands for IPython."""

import os
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython import get_ipython

from .ai_client import AIClient


@magics_class
class AIMagics(Magics):
    """AI-powered magic commands for IPython."""
    
    def __init__(self, shell=None):
        super().__init__(shell)
        self.ai_client = AIClient()
    
    @line_magic
    @magic_arguments()
    @argument('--execute', '-e', action='store_true', 
              help='Execute the generated code immediately')
    @argument('--model', '-m', default='claude-3-sonnet-20240229',
              help='AI model to use')
    def ai(self, line):
        """Generate Python code from natural language description.
        
        Usage:
            %ai create a pandas dataframe with sample data
            %ai -e plot a sine wave with matplotlib
        """
        args = parse_argstring(self.ai, line)
        description = args.line if hasattr(args, 'line') else line.strip()
        
        if not description:
            print("Please provide a description of what you want to do.")
            return
            
        try:
            # Generate code
            print(f"🤖 Generating code for: {description}")
            code = self.ai_client.generate_code(description, model=args.model)
            
            if not code:
                print("❌ No code generated. Please try rephrasing your request.")
                return
                
            # Display the generated code
            print(f"\n📝 Generated code:")
            print("```python")
            print(code)
            print("```")
            
            # Execute if requested
            if args.execute:
                print("\n🚀 Executing...")
                try:
                    self.shell.run_cell(code)
                except Exception as e:
                    print(f"❌ Execution error: {e}")
            else:
                print("\n💡 Use %ai -e to execute, or copy the code above.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    @line_magic
    def ai_question(self, line):
        """Ask the AI a question about your code or data.
        
        Usage:
            %ai? what does this variable contain
            %ai? how do I optimize this pandas operation
        """
        question = line.strip()
        
        if not question:
            print("Please ask a question.")
            return
            
        try:
            print(f"🤖 Thinking about: {question}")
            answer = self.ai_client.ask_question(question)
            print(f"\n💭 {answer}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
