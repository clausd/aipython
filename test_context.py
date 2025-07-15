"""Tests for AI context gathering and usage."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from aipython.ai_client import AIClient
from aipython.magic import AIMagics


class TestContextGathering:
    """Test context gathering functionality."""
    
    def test_get_ipython_context_no_ipython(self):
        """Test context gathering when not in IPython."""
        client = AIClient(api_key="test-key")
        
        with patch('IPython.get_ipython', return_value=None):
            context = client.get_ipython_context()
            assert context == {}
    
    def test_get_ipython_context_basic_variables(self):
        """Test context gathering with basic variables."""
        client = AIClient(api_key="test-key")
        
        # Mock IPython instance
        mock_ip = Mock()
        mock_ip.user_ns = {
            'x': 42,
            'name': 'test',
            'data': [1, 2, 3, 4, 5],
            '_private': 'hidden',
            'In': 'should_be_filtered',
            'Out': 'should_be_filtered'
        }
        mock_ip.history_manager.get_range.return_value = ['cmd1', 'cmd2', 'cmd3']
        
        with patch('IPython.get_ipython', return_value=mock_ip):
            context = client.get_ipython_context()
            
            # Check that context contains expected variables
            assert 'variables' in context
            assert 'history_length' in context
            
            variables = context['variables']
            assert 'x' in variables
            assert 'name' in variables
            assert 'data' in variables
            
            # Check that private variables are filtered out
            assert '_private' not in variables
            assert 'In' not in variables
            assert 'Out' not in variables
            
            # Check variable info
            assert variables['x']['type'] == 'int'
            assert variables['x']['preview'] == '42'
            assert variables['name']['type'] == 'str'
            assert variables['name']['preview'] == 'test'
            assert variables['data']['type'] == 'list'
            assert variables['data']['shape'] == '(len=5)'
            
            assert context['history_length'] == 3
    
    def test_get_ipython_context_pandas_numpy(self):
        """Test context gathering with pandas and numpy objects."""
        client = AIClient(api_key="test-key")
        
        # Create test data
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        series = pd.Series([1, 2, 3], name='test_series')
        array = np.array([[1, 2], [3, 4]])
        
        mock_ip = Mock()
        mock_ip.user_ns = {
            'df': df,
            'series': series,
            'array': array
        }
        mock_ip.history_manager.get_range.return_value = []
        
        with patch('IPython.get_ipython', return_value=mock_ip):
            context = client.get_ipython_context()
            
            variables = context['variables']
            
            # Check DataFrame info
            assert variables['df']['type'] == 'DataFrame'
            assert variables['df']['shape'] == '(3, 2)'
            assert 'columns:' in variables['df']['preview']
            assert 'A' in variables['df']['preview']
            assert 'B' in variables['df']['preview']
            
            # Check Series info
            assert variables['series']['type'] == 'Series'
            assert variables['series']['shape'] == '(3,)'
            
            # Check numpy array info
            assert variables['array']['type'] == 'ndarray'
            assert variables['array']['shape'] == '(2, 2)'
            assert 'dtype:' in variables['array']['preview']


class TestContextUsage:
    """Test that context is properly used in AI interactions."""
    
    @patch('aipython.ai_client.Client')
    def test_generate_code_with_context(self, mock_client_class):
        """Test that generate_code passes context to AI."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="x + 1")]
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = AIClient(api_key="test-key")
        
        # Test context
        context = {
            'variables': {
                'x': {'type': 'int', 'preview': '42'},
                'df': {'type': 'DataFrame', 'shape': '(100, 5)', 'preview': 'columns: [A, B, C, D, E]'}
            }
        }
        
        result = client.generate_code("increment x", context=context)
        
        # Verify client.chat was called
        assert mock_client.chat.called
        call_args = mock_client.chat.call_args[0][0]  # First positional arg (messages)
        
        # Check that context was included in user message
        user_message = call_args[1]['content']  # Second message should be user message
        assert 'Available variables in current session:' in user_message
        assert 'x: int = 42' in user_message
        assert 'df: DataFrame (100, 5) = columns: [A, B, C, D, E]' in user_message
        
        assert result == "x + 1"
    
    @patch('aipython.ai_client.Client')
    def test_ask_question_with_context(self, mock_client_class):
        """Test that ask_question passes context to AI."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="The variable x contains the integer 42")]
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        client = AIClient(api_key="test-key")
        
        # Test context
        context = {
            'variables': {
                'x': {'type': 'int', 'preview': '42'}
            }
        }
        
        result = client.ask_question("what is x?", context=context)
        
        # Verify client.chat was called
        assert mock_client.chat.called
        call_args = mock_client.chat.call_args[0][0]  # First positional arg (messages)
        
        # Check that context was included in user message
        user_message = call_args[1]['content']  # Second message should be user message
        assert 'Current session context:' in user_message
        assert 'x: int = 42' in user_message
        
        assert result == "The variable x contains the integer 42"


class TestMagicCommandsWithContext:
    """Test magic commands use context properly."""
    
    def test_ai_magic_calls_with_context(self):
        """Test that %ai magic command gathers and passes context."""
        # Create magic instance without shell (will be None)
        magic = AIMagics()
        
        # Mock the AI client methods
        magic.ai_client.get_ipython_context = Mock(return_value={
            'variables': {'x': {'type': 'int', 'preview': '42'}}
        })
        magic.ai_client.generate_code = Mock(return_value="print(x)")
        
        # Mock argparse
        with patch('aipython.magic.parse_argstring') as mock_parse:
            mock_args = Mock()
            mock_args.execute = False
            mock_args.model = 'claude-3-sonnet-20240229'
            mock_parse.return_value = mock_args
            
            # Call the magic command
            magic.ai("create a simple print statement")
            
            # Verify context was gathered and passed
            magic.ai_client.get_ipython_context.assert_called_once()
            # The first argument will be the line string, not args.line
            call_args = magic.ai_client.generate_code.call_args
            assert call_args[0][0] == "create a simple print statement"
            assert call_args[1]['model'] == 'claude-3-sonnet-20240229'
            assert call_args[1]['context'] == {'variables': {'x': {'type': 'int', 'preview': '42'}}}
    
    def test_ai_question_magic_calls_with_context(self):
        """Test that %ai? magic command gathers and passes context."""
        # Create magic instance without shell (will be None)
        magic = AIMagics()
        
        # Mock the AI client methods
        magic.ai_client.get_ipython_context = Mock(return_value={
            'variables': {'df': {'type': 'DataFrame', 'shape': '(100, 5)'}}
        })
        magic.ai_client.ask_question = Mock(return_value="It's a DataFrame with 100 rows and 5 columns")
        
        # Call the magic command
        magic.ai_question("what is df?")
        
        # Verify context was gathered and passed
        magic.ai_client.get_ipython_context.assert_called_once()
        magic.ai_client.ask_question.assert_called_once_with(
            "what is df?",
            context={'variables': {'df': {'type': 'DataFrame', 'shape': '(100, 5)'}}}
        )


if __name__ == '__main__':
    pytest.main([__file__])
