"""
Test module for veracity auditing functionality.

This module tests the integration between the VeracityAuditor and the
model wrapper system to ensure accurate quality assessment of model outputs.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from app.audit.veracity import VeracityAuditor
from app.services.models.wrapper_base import BaseModelWrapper, ModelInput, VeracityMetrics

# Test data
TEST_TRANSLATIONS = [
    {
        "source": "Hello, how are you today?",
        "target": "Hola, ¿cómo estás hoy?",
        "source_lang": "en",
        "target_lang": "es",
        "expect_verified": True
    },
    {
        "source": "I have 5 apples and 3 oranges.",
        "target": "Tengo 5 manzanas y 3 naranjas.",
        "source_lang": "en",
        "target_lang": "es",
        "expect_verified": True
    },
    {
        "source": "The red house costs $500,000.",
        "target": "La casa roja cuesta $.",  # Missing numbers
        "source_lang": "en",
        "target_lang": "es",
        "expect_verified": False
    },
    {
        "source": "How many people attended the conference?",
        "target": "How many people attended the conference?",  # Untranslated
        "source_lang": "en",
        "target_lang": "es",
        "expect_verified": False
    }
]

TEST_SIMPLIFICATIONS = [
    {
        "source": "The contingent liability associated with the litigation was not recognized in the financial statements as management believes it is not probable that an outflow of resources will be required to settle the obligation.",
        "target": "The company didn't record the lawsuit cost in their financial reports because they think they won't have to pay anything.",
        "lang": "en",
        "expect_verified": True
    },
    {
        "source": "The synthesis of the compound via the catalytic hydrogenation process resulted in a 95% yield with minimal by-products detected in the spectroscopic analysis.",
        "target": "Making the compound using catalytic hydrogenation gave a 95% yield with few by-products found in testing.",
        "lang": "en",
        "expect_verified": True
    },
    {
        "source": "The implementation of the proposed algorithm demonstrated a significant reduction in computational complexity while maintaining the accuracy of the results.",
        "target": "The implementation of the proposed algorithm demonstrated a significant reduction in computational complexity while maintaining the accuracy of the results.",  # Not simplified
        "lang": "en",
        "expect_verified": False
    }
]

# Mock model wrapper implementation for testing
class MockModelWrapper(BaseModelWrapper):
    def __init__(self):
        # Bypass original init
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.config = {}
        self.device = "cpu"
        self.execution_count = 0
        self.error_count = 0
        self.total_execution_time = 0
        self.max_execution_time = 0
        self.veracity_checker = None

    def _preprocess(self, input_data):
        return {"processed": input_data.text}
    
    def _run_inference(self, preprocessed):
        return "Mock model output"
    
    def _postprocess(self, model_output, input_data):
        from app.services.models.wrapper_base import ModelOutput
        return ModelOutput(result=model_output)

@pytest.fixture
def veracity_auditor():
    """Create a VeracityAuditor instance for testing."""
    auditor = VeracityAuditor()
    return auditor

@pytest.fixture
def mock_wrapper():
    """Create a mock model wrapper for testing."""
    wrapper = MockModelWrapper()
    return wrapper

@pytest.mark.asyncio
async def test_veracity_auditor_initialization(veracity_auditor):
    """Test that the VeracityAuditor initializes correctly."""
    assert veracity_auditor is not None
    assert veracity_auditor.enabled is True
    
    # Initialize the auditor
    await veracity_auditor.initialize()
    assert len(veracity_auditor.quality_statistics) == 0

@pytest.mark.asyncio
async def test_translation_verification(veracity_auditor):
    """Test verification of translations."""
    for test_case in TEST_TRANSLATIONS:
        result = await veracity_auditor.verify_translation(
            test_case["source"],
            test_case["target"],
            test_case["source_lang"],
            test_case["target_lang"]
        )
        
        assert "verified" in result
        assert "score" in result
        assert "confidence" in result
        assert "issues" in result
        
        if test_case["expect_verified"]:
            assert result["verified"] is True, f"Expected verification to pass for: {test_case}"
        else:
            # Either verified is False or there are critical issues
            critical_issues = [i for i in result["issues"] if i["severity"] == "critical"]
            assert result["verified"] is False or len(critical_issues) > 0, \
                f"Expected verification to fail for: {test_case}"

@pytest.mark.asyncio
async def test_simplification_verification(veracity_auditor):
    """Test verification of text simplifications."""
    for test_case in TEST_SIMPLIFICATIONS:
        result = await veracity_auditor._verify_simplification(
            test_case["source"],
            test_case["target"],
            test_case["lang"]
        )
        
        assert "verified" in result
        assert "score" in result
        assert "confidence" in result
        assert "issues" in result
        
        if test_case["expect_verified"]:
            assert result["verified"] is True, f"Expected verification to pass for: {test_case}"
        else:
            # Either verified is False or there are critical issues
            critical_issues = [i for i in result["issues"] if i["severity"] == "critical"]
            assert result["verified"] is False or len(critical_issues) > 0, \
                f"Expected verification to fail for: {test_case}"

@pytest.mark.asyncio
async def test_check_method_routing(veracity_auditor):
    """Test that the generic check method routes to the correct specialized method."""
    # Test translation routing
    translation_options = {
        "source_language": "en",
        "target_language": "es",
        "operation": "translation"
    }
    
    with patch.object(veracity_auditor, 'verify_translation') as mock_verify_translation:
        mock_verify_translation.return_value = {"verified": True}
        result = await veracity_auditor.check("source", "target", translation_options)
        mock_verify_translation.assert_called_once()
        assert result == {"verified": True}
    
    # Test simplification routing
    simplification_options = {
        "source_language": "en",
        "operation": "simplification"
    }
    
    with patch.object(veracity_auditor, '_verify_simplification') as mock_verify_simplification:
        mock_verify_simplification.return_value = {"verified": True}
        result = await veracity_auditor.check("source", "target", simplification_options)
        mock_verify_simplification.assert_called_once()
        assert result == {"verified": True}

@pytest.mark.asyncio
async def test_integration_with_wrapper(mock_wrapper, veracity_auditor):
    """Test integration between the veracity auditor and model wrapper."""
    # Set up the mock wrapper with our veracity auditor
    mock_wrapper.veracity_checker = veracity_auditor
    
    # Test the async veracity check method
    input_data = ModelInput(
        text="Hello, how are you?",
        source_language="en",
        target_language="es"
    )
    
    result = await mock_wrapper._check_veracity("Hola, ¿cómo estás?", input_data)
    
    assert isinstance(result, VeracityMetrics)
    assert hasattr(result, "score")
    assert hasattr(result, "confidence")
    assert hasattr(result, "checks_passed")
    assert hasattr(result, "checks_failed")
    assert hasattr(result, "warnings")
    
    # Test the sync veracity check method
    result = mock_wrapper._check_veracity_sync("Hola, ¿cómo estás?", input_data)
    
    assert isinstance(result, VeracityMetrics)
    assert hasattr(result, "score")
    assert hasattr(result, "confidence")
    assert hasattr(result, "checks_passed")
    assert hasattr(result, "checks_failed")
    assert hasattr(result, "warnings")

def test_quality_statistics(veracity_auditor):
    """Test that quality statistics are properly tracked."""
    # Add some test data to quality statistics
    veracity_auditor.quality_statistics = {
        "en-es": {
            "verified_count": 8,
            "total_count": 10,
            "average_score": 0.85,
            "average_confidence": 0.9,
            "issue_counts": {"missing_numbers": 2}
        },
        "en-fr": {
            "verified_count": 7,
            "total_count": 9,
            "average_score": 0.8,
            "average_confidence": 0.85,
            "issue_counts": {"length_mismatch": 1, "low_semantic_similarity": 1}
        }
    }
    
    stats = veracity_auditor.get_quality_statistics()
    
    assert "overall" in stats
    assert "by_language_pair" in stats
    assert stats["overall"]["total_count"] == 19
    assert stats["overall"]["verified_count"] == 15
    assert abs(stats["overall"]["verification_rate"] - (15/19)) < 0.001
    assert "top_issues" in stats["overall"]
    assert len(stats["by_language_pair"]) == 2

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])