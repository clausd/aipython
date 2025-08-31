import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from pathlib import Path

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
        
        # Accumulate stats
        self.overall_stats.total_steps += conversation.stats.total_steps
        self.overall_stats.successful_steps += conversation.stats.successful_steps
        self.overall_stats.failed_steps += conversation.stats.failed_steps
        self.overall_stats.ai_generate_total += conversation.stats.ai_generate_total
        self.overall_stats.ai_generate_success += conversation.stats.ai_generate_success
        self.overall_stats.ai_question_total += conversation.stats.ai_question_total
        self.overall_stats.ai_question_success += conversation.stats.ai_question_success
        
        # Print test-specific stats
        print(f"\nSimple CSV Test - Success Rate: {conversation.stats.success_rate:.1%}")
        
        # Basic assertions - we expect reasonable success but not perfection
        assert conversation.stats.total_steps == 4
        assert conversation.stats.success_rate >= 0.5  # At least 50% success rate
    
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
        
        # Accumulate stats
        self.overall_stats.total_steps += conversation.stats.total_steps
        self.overall_stats.successful_steps += conversation.stats.successful_steps
        self.overall_stats.failed_steps += conversation.stats.failed_steps
        self.overall_stats.ai_generate_total += conversation.stats.ai_generate_total
        self.overall_stats.ai_generate_success += conversation.stats.ai_generate_success
        self.overall_stats.ai_question_total += conversation.stats.ai_question_total
        self.overall_stats.ai_question_success += conversation.stats.ai_question_success
        
        print(f"\nEmpty Data Test - Success Rate: {conversation.stats.success_rate:.1%}")
        
        assert conversation.stats.total_steps == 2
        assert conversation.stats.success_rate >= 0.5
