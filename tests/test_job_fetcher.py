"""
Unit tests for Job Fetcher
"""

import pytest
from tools.job_fetcher import JobFetcher


class TestJobFetcher:
    """Test cases for JobFetcher class"""
    
    @pytest.fixture
    def fetcher(self):
        """Create a JobFetcher instance for testing"""
        return JobFetcher()
    
    def test_initialization(self, fetcher):
        """Test that JobFetcher initializes correctly"""
        assert fetcher is not None
        assert hasattr(fetcher, 'jsearch_url')
        assert hasattr(fetcher, 'adzuna_url')
        assert fetcher.min_request_interval == 1
    
    def test_rate_limiting(self, fetcher):
        """Test that rate limiting works"""
        import time
        start = time.time()
        fetcher._rate_limit()
        fetcher._rate_limit()
        elapsed = time.time() - start
        # Should take at least 1 second due to rate limiting
        assert elapsed >= 1.0
    
    def test_parse_jsearch_job(self, fetcher):
        """Test parsing JSearch API response"""
        sample_job = {
            'job_id': '123',
            'job_title': 'Software Engineer',
            'employer_name': 'Tech Corp',
            'job_city': 'San Francisco',
            'job_description': 'Great opportunity',
            'job_highlights': {
                'Qualifications': ['Python', 'Django'],
                'Responsibilities': ['Develop features'],
                'Benefits': ['Health insurance']
            },
            'job_employment_type': 'FULLTIME',
            'job_apply_link': 'https://example.com'
        }
        
        parsed = fetcher._parse_jsearch_job(sample_job)
        
        assert parsed['id'] == '123'
        assert parsed['title'] == 'Software Engineer'
        assert parsed['company'] == 'Tech Corp'
        assert parsed['location'] == 'San Francisco'
        assert parsed['source'] == 'JSearch'
        assert len(parsed['requirements']) == 2
    
    def test_parse_adzuna_job(self, fetcher):
        """Test parsing Adzuna API response"""
        sample_job = {
            'id': '456',
            'title': 'Data Scientist',
            'company': {'display_name': 'Data Inc'},
            'location': {'display_name': 'New York'},
            'description': 'Exciting role',
            'contract_type': 'permanent',
            'redirect_url': 'https://example.com'
        }
        
        parsed = fetcher._parse_adzuna_job(sample_job)
        
        assert parsed['id'] == '456'
        assert parsed['title'] == 'Data Scientist'
        assert parsed['company'] == 'Data Inc'
        assert parsed['location'] == 'New York'
        assert parsed['source'] == 'Adzuna'
    
    def test_missing_api_keys(self, fetcher, monkeypatch):
        """Test behavior when API keys are missing"""
        # Remove API keys
        monkeypatch.setattr(fetcher, 'rapidapi_key', None)
        monkeypatch.setattr(fetcher, 'adzuna_app_id', None)
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            fetcher.search_jobs_jsearch("Python Developer")
        
        with pytest.raises(ValueError):
            fetcher.search_jobs_adzuna("Python Developer")
    
    def test_search_jobs_with_defaults(self, fetcher):
        """Test search_jobs with default parameters"""
        # This test would require actual API keys and network access
        # For now, we just verify the method exists and has correct signature
        assert hasattr(fetcher, 'search_jobs')
        
        # Test with mock to avoid actual API call
        import unittest.mock as mock
        with mock.patch.object(fetcher, 'search_jobs_jsearch', return_value=[]):
            result = fetcher.search_jobs("Software Engineer")
            assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
