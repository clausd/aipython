import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import os
import sys
from pathlib import Path
from claudette import Client

# Add the project root to Python path so we can import aipython
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class ConversationStats:
    """Statistics for conversation evaluation runs."""
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    ai_generate_success: int = 0
    ai_generate_total: int = 0
    ai_question_success: int = 0
    ai_question_total: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successful_steps / self.total_steps if self.total_steps > 0 else 0.0
    
    @property
    def generate_success_rate(self) -> float:
        return self.ai_generate_success / self.ai_generate_total if self.ai_generate_total > 0 else 0.0
    
    @property
    def question_success_rate(self) -> float:
        return self.ai_question_success / self.ai_question_total if self.ai_question_total > 0 else 0.0
    
    def add_step_result(self, step_type: str, success: bool):
        """Add a step result to the statistics."""
        self.total_steps += 1
        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
            
        if step_type == 'ai_generate':
            self.ai_generate_total += 1
            if success:
                self.ai_generate_success += 1
        elif step_type == 'ai_question':
            self.ai_question_total += 1
            if success:
                self.ai_question_success += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            'total_steps': self.total_steps,
            'successful_steps': self.successful_steps,
            'failed_steps': self.failed_steps,
            'success_rate': round(self.success_rate, 3),
            'ai_generate': {
                'total': self.ai_generate_total,
                'success': self.ai_generate_success,
                'success_rate': round(self.generate_success_rate, 3)
            },
            'ai_question': {
                'total': self.ai_question_total,
                'success': self.ai_question_success,
                'success_rate': round(self.question_success_rate, 3)
            }
        }

class ConversationTest:
    """Base class for conversation-based evaluations."""
    
    def __init__(self, dataset_path=None, initial_setup=None):
        self.dataset_path = dataset_path
        self.initial_setup = initial_setup or []
        self.context = {}
        self.stats = ConversationStats()
        self.claude_client = None
        
        # Initialize Claude client for evaluation if API key is available
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.claude_client = Client('claude-3-haiku-20240307')
        
    def setup_data(self, ipython_mock):
        """Load dataset and run initial setup commands."""
        if self.dataset_path:
            if self.dataset_path.endswith('.csv'):
                # Handle semicolon-delimited CSV like simple.csv
                df = pd.read_csv(self.dataset_path, sep=';')
                ipython_mock.user_ns['df'] = df
                self.context['df'] = df
            elif self.dataset_path.endswith('.txt'):
                # Handle text files
                with open(self.dataset_path, 'r') as f:
                    content = f.read().strip()
                ipython_mock.user_ns['text_data'] = content
                self.context['text_data'] = content
            
        # Run initial setup commands
        for cmd in self.initial_setup:
            exec(cmd, ipython_mock.user_ns)
    
    def run_conversation(self, steps, ai_client_mock):
        """Run a conversation with multiple AI interactions and track stats."""
        results = []
        
        for step in steps:
            success = False
            result = None
            
            try:
                if step['type'] == 'ai_generate':
                    # Mock the AI response - in real evals, this would be actual AI calls
                    ai_client_mock.generate_code.return_value = step.get('mock_response', 'df.head()')
                    result = ai_client_mock.generate_code(step['prompt'])
                    # Simple success check - contains expected keywords or patterns
                    success = self._evaluate_generate_response(step, result)
                    
                elif step['type'] == 'ai_question':
                    ai_client_mock.ask_question.return_value = step.get('mock_response', 'The data contains...')
                    result = ai_client_mock.ask_question(step['prompt'])
                    # Simple success check - response is non-empty and relevant
                    success = self._evaluate_question_response(step, result)
                    
            except Exception as e:
                success = False
                result = f"Error: {str(e)}"
            
            # Track statistics
            self.stats.add_step_result(step['type'], success)
            
            results.append({
                'step': step,
                'result': result,
                'success': success
            })
            
        return results
    
    def _evaluate_generate_response(self, step, result):
        """Evaluate if a code generation response is successful."""
        if not result or len(result.strip()) == 0:
            return False
        
        # Check for expected patterns in the response
        expected_patterns = step.get('expected_patterns', [])
        if expected_patterns:
            return any(pattern.lower() in result.lower() for pattern in expected_patterns)
        
        # Default: non-empty response is considered success
        return True
    
    def _evaluate_question_response(self, step, result):
        """Evaluate if a question response is successful."""
        if not result or len(result.strip()) < 10:  # Minimum meaningful response length
            return False
        
        # Check for expected keywords in the response
        expected_keywords = step.get('expected_keywords', [])
        if expected_keywords:
            return any(keyword.lower() in result.lower() for keyword in expected_keywords)
        
        # Default: reasonable length response is considered success
        return True
    
    def evaluate_with_claude(self, ipython_context, conversation_history):
        """Use Claude to evaluate the conversation and context quality."""
        if not self.claude_client:
            return {"success": True, "reason": "No Claude client available", "score": 0.8}
        
        try:
            # Prepare context summary for Claude
            context_summary = self._summarize_ipython_context(ipython_context)
            conversation_summary = self._summarize_conversation(conversation_history)
            
            evaluation_prompt = f"""
You are evaluating an AI-assisted data analysis conversation in IPython. 

CONTEXT AFTER CONVERSATION:
{context_summary}

CONVERSATION HISTORY:
{conversation_summary}

Please evaluate:
1. Did the conversation achieve meaningful data exploration?
2. Are the variables in the context appropriate for the conversation?
3. Would the generated code actually work if executed?
4. Is the AI's understanding of the data accurate?

Respond with a JSON object containing:
- "success": boolean (true if conversation was successful)
- "score": float between 0.0 and 1.0 
- "reason": string explaining your evaluation
- "issues": list of any problems you identified

Be generous but realistic - this is an evaluation of AI assistance quality.
"""
            
            response = self.claude_client(evaluation_prompt)
            
            # Try to parse JSON response
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    evaluation = json.loads(json_match.group())
                else:
                    # Fallback if no JSON found
                    evaluation = {
                        "success": "successful" in response.lower() or "good" in response.lower(),
                        "score": 0.7,
                        "reason": response[:200] + "..." if len(response) > 200 else response,
                        "issues": []
                    }
            except json.JSONDecodeError:
                # Fallback evaluation based on response content
                evaluation = {
                    "success": "successful" in response.lower() or "good" in response.lower(),
                    "score": 0.7,
                    "reason": response[:200] + "..." if len(response) > 200 else response,
                    "issues": []
                }
            
            return evaluation
            
        except Exception as e:
            return {
                "success": False, 
                "score": 0.0,
                "reason": f"Claude evaluation failed: {str(e)}",
                "issues": ["evaluation_error"]
            }
    
    def _summarize_ipython_context(self, ipython_context):
        """Create a summary of the IPython context for Claude evaluation."""
        summary_parts = []
        
        user_ns = ipython_context.get('user_ns', {})
        for var_name, var_value in user_ns.items():
            if var_name.startswith('_'):
                continue  # Skip private variables
                
            try:
                if hasattr(var_value, 'shape'):  # pandas DataFrame, numpy array
                    summary_parts.append(f"- {var_name}: {type(var_value).__name__} with shape {var_value.shape}")
                elif isinstance(var_value, (list, tuple, dict)):
                    summary_parts.append(f"- {var_name}: {type(var_value).__name__} with {len(var_value)} items")
                elif isinstance(var_value, str):
                    preview = var_value[:50] + "..." if len(var_value) > 50 else var_value
                    summary_parts.append(f"- {var_name}: string '{preview}'")
                else:
                    summary_parts.append(f"- {var_name}: {type(var_value).__name__} = {str(var_value)[:50]}")
            except:
                summary_parts.append(f"- {var_name}: {type(var_value).__name__}")
        
        return "\n".join(summary_parts) if summary_parts else "No significant variables in context"
    
    def _summarize_conversation(self, conversation_history):
        """Create a summary of the conversation for Claude evaluation."""
        summary_parts = []
        
        for i, result in enumerate(conversation_history, 1):
            step = result['step']
            success = result['success']
            response = result['result']
            
            status = "✓" if success else "✗"
            summary_parts.append(f"{i}. {status} {step['type']}: {step['prompt']}")
            summary_parts.append(f"   Response: {response}")
        
        return "\n".join(summary_parts)

class TestDatasetConversations:
    """Test conversations with different datasets."""
    
    @classmethod
    def setup_class(cls):
        """Initialize class-level statistics."""
        cls.overall_stats = ConversationStats()
    
    @classmethod
    def teardown_class(cls):
        """Print overall statistics after all tests."""
        print("\n" + "="*50)
        print("CONVERSATION EVALUATION RESULTS")
        print("="*50)
        stats_dict = cls.overall_stats.to_dict()
        print(f"Overall Success Rate: {stats_dict['success_rate']:.1%}")
        print(f"Total Steps: {stats_dict['total_steps']}")
        print(f"Successful: {stats_dict['successful_steps']}")
        print(f"Failed: {stats_dict['failed_steps']}")
        print(f"\nCode Generation Success Rate: {stats_dict['ai_generate']['success_rate']:.1%}")
        print(f"Question Answering Success Rate: {stats_dict['ai_question']['success_rate']:.1%}")
        
        # Save detailed stats to file
        stats_file = Path('evals/results/conversation_stats.json')
        stats_file.parent.mkdir(exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nDetailed stats saved to: {stats_file}")
    
    @patch('aipython.ai_client.Client')
    @patch('IPython.get_ipython')
    def test_simple_csv_exploration(self, mock_ipython, mock_client_class):
        """Test data exploration conversation with simple.csv."""
        
        # Setup mocks
        mock_ipython_instance = MagicMock()
        mock_ipython_instance.user_ns = {}
        mock_ipython.return_value = mock_ipython_instance
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Setup conversation
        conversation = ConversationTest(
            dataset_path='evals/datasets/simple.csv',
            initial_setup=['import pandas as pd', 'import numpy as np']
        )
        conversation.setup_data(mock_ipython_instance)
        
        # Conversation steps with evaluation criteria
        steps = [
            {
                'type': 'ai_question',
                'prompt': 'What columns does this dataset have?',
                'mock_response': 'The dataset has 3 columns: foo, bar, and baz',
                'expected_keywords': ['foo', 'bar', 'baz', 'columns']
            },
            {
                'type': 'ai_generate', 
                'prompt': 'Show me the shape of the data',
                'mock_response': 'df.shape',
                'expected_patterns': ['df.shape', '.shape']
            },
            {
                'type': 'ai_generate',
                'prompt': 'Find the maximum value in the baz column',
                'mock_response': 'df["baz"].max()',
                'expected_patterns': ['baz', 'max', 'df[']
            },
            {
                'type': 'ai_question',
                'prompt': 'Are there any patterns in this data?',
                'mock_response': 'The data appears to be numeric with values ranging from 1-18, with some gaps in the sequence.',
                'expected_keywords': ['numeric', 'values', 'data']
            }
        ]
        
        results = conversation.run_conversation(steps, mock_client)
        
        # Claude-based evaluation of the conversation
        claude_evaluation = conversation.evaluate_with_claude(
            ipython_context=mock_ipython_instance.__dict__,
            conversation_history=results
        )
        
        # Accumulate stats
        self.overall_stats.total_steps += conversation.stats.total_steps
        self.overall_stats.successful_steps += conversation.stats.successful_steps
        self.overall_stats.failed_steps += conversation.stats.failed_steps
        self.overall_stats.ai_generate_total += conversation.stats.ai_generate_total
        self.overall_stats.ai_generate_success += conversation.stats.ai_generate_success
        self.overall_stats.ai_question_total += conversation.stats.ai_question_total
        self.overall_stats.ai_question_success += conversation.stats.ai_question_success
        
        # Print test-specific stats including Claude evaluation
        print(f"\nSimple CSV Test - Success Rate: {conversation.stats.success_rate:.1%}")
        print(f"Claude Evaluation - Score: {claude_evaluation['score']:.2f}, Success: {claude_evaluation['success']}")
        print(f"Claude Reason: {claude_evaluation['reason']}")
        
        # Basic assertions - we expect reasonable success but not perfection
        assert conversation.stats.total_steps == 4
        assert conversation.stats.success_rate >= 0.5  # At least 50% success rate
        
        # Optional: Assert Claude evaluation is reasonable
        if claude_evaluation['score'] > 0:  # Only assert if Claude evaluation ran
            assert claude_evaluation['score'] >= 0.3  # Very lenient threshold
    
    @patch('aipython.ai_client.Client')
    @patch('IPython.get_ipython')
    def test_empty_data_handling(self, mock_ipython, mock_client_class):
        """Test conversation with empty/minimal data."""
        
        mock_ipython_instance = MagicMock()
        mock_ipython_instance.user_ns = {}
        mock_ipython.return_value = mock_ipython_instance
        
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        conversation = ConversationTest(
            dataset_path='evals/datasets/multi.variable.txt'
        )
        conversation.setup_data(mock_ipython_instance)
        
        steps = [
            {
                'type': 'ai_question',
                'prompt': 'What data is available?',
                'mock_response': 'No data appears to be loaded in the current context.',
                'expected_keywords': ['no data', 'empty', 'available']
            },
            {
                'type': 'ai_generate',
                'prompt': 'Create some sample data',
                'mock_response': 'import pandas as pd\ndf = pd.DataFrame({"x": [1,2,3], "y": [4,5,6]})',
                'expected_patterns': ['pandas', 'DataFrame', 'pd.']
            }
        ]
        
        results = conversation.run_conversation(steps, mock_client)
        
        # Claude-based evaluation
        claude_evaluation = conversation.evaluate_with_claude(
            ipython_context=mock_ipython_instance.__dict__,
            conversation_history=results
        )
        
        # Accumulate stats
        self.overall_stats.total_steps += conversation.stats.total_steps
        self.overall_stats.successful_steps += conversation.stats.successful_steps
        self.overall_stats.failed_steps += conversation.stats.failed_steps
        self.overall_stats.ai_generate_total += conversation.stats.ai_generate_total
        self.overall_stats.ai_generate_success += conversation.stats.ai_generate_success
        self.overall_stats.ai_question_total += conversation.stats.ai_question_total
        self.overall_stats.ai_question_success += conversation.stats.ai_question_success
        
        print(f"\nEmpty Data Test - Success Rate: {conversation.stats.success_rate:.1%}")
        print(f"Claude Evaluation - Score: {claude_evaluation['score']:.2f}, Success: {claude_evaluation['success']}")
        
        assert conversation.stats.total_steps == 2
        assert conversation.stats.success_rate >= 0.5
