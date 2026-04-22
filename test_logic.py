import unittest
from unittest.mock import patch, MagicMock
import os
import json

import logic

class TestGenAIIntegration(unittest.TestCase):
    
    def setUp(self):
        os.environ["PURDUE_GENAI_API_KEY"] = "mock-key"
    
    @patch('logic.requests.post')
    def test_valid_openai_style_dict(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": json.dumps({
                    "urgency_label": "routine",
                    "type_label": "admin",
                    "route_label": "front desk",
                    "confidence": 0.95,
                    "draft_response": "Hello, thank you.",
                    "rationale": "Clear admin request."
                })}}
            ]
        }
        mock_post.return_value = mock_resp
        
        result = logic.call_purdue_genai("I need a form.")
        self.assertEqual(result["status"], "Success")
        self.assertEqual(result["data"]["urgency_label"], "routine")
        
    @patch('logic.requests.post')
    def test_list_json_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        # API unexpectedly returns a list
        mock_resp.json.return_value = [{"urgency_label": "routine"}]
        mock_post.return_value = mock_resp
        
        result = logic.call_purdue_genai("I need a form.")
        self.assertEqual(result["status"], "InvalidAPIResponse")
        self.assertIn("Type: list", result["error"])
        
    @patch('logic.requests.post')
    def test_string_json_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        # API unexpectedly returns a raw primitive string
        mock_resp.json.return_value = "This is a string not a dictionary payload"
        mock_post.return_value = mock_resp
        
        result = logic.call_purdue_genai("I need a form.")
        self.assertEqual(result["status"], "InvalidAPIResponse")
        self.assertIn("Type: str", result["error"])
        self.assertIn("This is a string", result["error"])
        
    @patch('logic.requests.post')
    def test_non_json_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.text = "<html>Bad Gateway</html>"
        mock_resp.headers = {"Content-Type": "text/html"}
        # Simulate .json() failing
        mock_resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
        mock_post.return_value = mock_resp
        
        result = logic.call_purdue_genai("I need a form.")
        self.assertEqual(result["status"], "InvalidAPIResponse")
        self.assertIn("Bad Gateway", result["error"])
        self.assertIn("Status: 502", result["error"])
        
    @patch('logic.requests.post')
    def test_empty_choices_list(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": []
        }
        mock_post.return_value = mock_resp
        
        result = logic.call_purdue_genai("I need a form.")
        self.assertEqual(result["status"], "EmptyResponse")
        self.assertIn("no choices", result["error"])

if __name__ == '__main__':
    unittest.main()
