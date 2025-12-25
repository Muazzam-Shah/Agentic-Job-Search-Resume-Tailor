"""
Job Data Fetcher - Retrieves job postings from free APIs
Supports JSearch (RapidAPI) and Adzuna APIs
"""

import os
import requests
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class JobFetcher:
    """Fetches job data from various free job APIs"""
    
    def __init__(self):
        self.rapidapi_key = os.getenv('RAPIDAPI_KEY')
        self.adzuna_app_id = os.getenv('ADZUNA_APP_ID')
        self.adzuna_app_key = os.getenv('ADZUNA_APP_KEY')
        
        # API endpoints
        self.jsearch_url = "https://jsearch.p.rapidapi.com/search"
        self.adzuna_url = "https://api.adzuna.com/v1/api/jobs"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1  # seconds between requests
        
    def _rate_limit(self):
        """Implement rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def search_jobs_jsearch(
        self, 
        query: str, 
        location: str = None,
        num_pages: int = 1
    ) -> List[Dict]:
        """
        Search jobs using JSearch API (RapidAPI)
        
        Args:
            query: Job search query (e.g., "Software Engineer")
            location: Location to search (e.g., "Pakistan", "London", "Remote")
            num_pages: Number of pages to fetch (default: 1, each page ~10 jobs)
            
        Returns:
            List of job dictionaries with structured data
        """
        if not self.rapidapi_key:
            raise ValueError("RAPIDAPI_KEY not found in environment variables")
        
        self._rate_limit()
        
        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        
        # Combine query and location for better JSearch accuracy
        search_query = query
        if location:
            search_query = f"{query} in {location}"
        
        params = {
            "query": search_query,
            "page": "1",
            "num_pages": str(num_pages),
            "date_posted": "all"
        }
        
        try:
            response = requests.get(
                self.jsearch_url,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            jobs = data.get('data', [])
            print(f"JSearch API returned {len(jobs)} jobs")
            parsed_jobs = [self._parse_jsearch_job(job) for job in jobs]
            print(f"Successfully parsed {len(parsed_jobs)} jobs")
            return parsed_jobs
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from JSearch API: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_jobs_adzuna(
        self,
        query: str,
        location: str = "us",
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search jobs using Adzuna API (fallback)
        
        Args:
            query: Job search query
            location: Country code (default: "us")
            max_results: Maximum number of results (default: 10)
            
        Returns:
            List of job dictionaries with structured data
        """
        if not self.adzuna_app_id or not self.adzuna_app_key:
            raise ValueError("ADZUNA_APP_ID or ADZUNA_APP_KEY not found")
        
        self._rate_limit()
        
        url = f"{self.adzuna_url}/{location}/search/1"
        
        params = {
            "app_id": self.adzuna_app_id,
            "app_key": self.adzuna_app_key,
            "results_per_page": max_results,
            "what": query,
            "content-type": "application/json"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            jobs = data.get('results', [])
            return [self._parse_adzuna_job(job) for job in jobs]
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Adzuna API: {e}")
            return []
    
    def search_jobs(
        self,
        query: str,
        location: str = None,
        max_results: int = 10,
        prefer_api: str = "jsearch"
    ) -> List[Dict]:
        """
        Search jobs with automatic fallback between APIs
        
        Args:
            query: Job search query
            location: Location to search (city, country, or None for global)
            max_results: Maximum results to return
            prefer_api: Preferred API ("jsearch" or "adzuna")
            
        Returns:
            List of structured job data
        """
        jobs = []
        
        # Try preferred API first
        if prefer_api == "jsearch" and self.rapidapi_key:
            jobs = self.search_jobs_jsearch(query, location)
        elif prefer_api == "adzuna" and self.adzuna_app_id:
            # For Adzuna, convert location to country code if needed
            adzuna_location = self._convert_to_country_code(location) if location else "us"
            jobs = self.search_jobs_adzuna(query, adzuna_location, max_results)
        
        # Fallback to alternative API if primary fails
        if not jobs:
            if prefer_api == "jsearch" and self.adzuna_app_id:
                print("Falling back to Adzuna API...")
                jobs = self.search_jobs_adzuna(query, location.lower(), max_results)
            elif prefer_api == "adzuna" and self.rapidapi_key:
                print("Falling back to JSearch API...")
                jobs = self.search_jobs_jsearch(query, location)
        
        return jobs[:max_results]
    
    def _convert_to_country_code(self, location: str) -> str:
        """
        Convert location name to ISO country code for Adzuna API
        
        Args:
            location: Location name (e.g., "Pakistan", "United Kingdom")
            
        Returns:
            2-letter country code (e.g., "pk", "gb")
        """
        # Common location to country code mappings
        location_map = {
            'pakistan': 'pk',
            'india': 'in',
            'united states': 'us',
            'us': 'us',
            'usa': 'us',
            'united kingdom': 'gb',
            'uk': 'gb',
            'canada': 'ca',
            'australia': 'au',
            'germany': 'de',
            'france': 'fr',
            'spain': 'es',
            'italy': 'it',
            'netherlands': 'nl',
            'poland': 'pl',
            'brazil': 'br',
            'mexico': 'mx',
            'singapore': 'sg',
            'new zealand': 'nz',
            'south africa': 'za'
        }
        
        location_lower = location.lower().strip()
        return location_map.get(location_lower, 'us')  # Default to US if not found
    
    def _parse_jsearch_job(self, job: Dict) -> Dict:
        """Parse JSearch API response into standard format"""
        return {
            'id': job.get('job_id'),
            'title': job.get('job_title'),
            'company': job.get('employer_name'),
            'location': job.get('job_city') or job.get('job_state') or job.get('job_country'),
            'description': job.get('job_description', ''),
            'requirements': job.get('job_highlights', {}).get('Qualifications', []),
            'responsibilities': job.get('job_highlights', {}).get('Responsibilities', []),
            'benefits': job.get('job_highlights', {}).get('Benefits', []),
            'employment_type': job.get('job_employment_type'),
            'posted_date': job.get('job_posted_at_datetime_utc'),
            'salary_min': job.get('job_min_salary'),
            'salary_max': job.get('job_max_salary'),
            'apply_link': job.get('job_apply_link'),
            'source': 'JSearch',
            'raw_data': job
        }
    
    def _parse_adzuna_job(self, job: Dict) -> Dict:
        """Parse Adzuna API response into standard format"""
        return {
            'id': job.get('id'),
            'title': job.get('title'),
            'company': job.get('company', {}).get('display_name'),
            'location': job.get('location', {}).get('display_name'),
            'description': job.get('description', ''),
            'requirements': [],  # Adzuna doesn't separate requirements
            'responsibilities': [],
            'benefits': [],
            'employment_type': job.get('contract_type'),
            'posted_date': job.get('created'),
            'salary_min': job.get('salary_min'),
            'salary_max': job.get('salary_max'),
            'apply_link': job.get('redirect_url'),
            'source': 'Adzuna',
            'raw_data': job
        }
    
    def get_job_by_id(self, job_id: str, api: str = "jsearch") -> Optional[Dict]:
        """
        Fetch detailed job information by ID
        
        Args:
            job_id: Job ID from previous search
            api: API to use ("jsearch" or "adzuna")
            
        Returns:
            Detailed job data or None
        """
        # This would require the job details endpoint
        # For now, return None - can be implemented if needed
        print(f"Job details endpoint not yet implemented for {api}")
        return None


if __name__ == "__main__":
    # Test the job fetcher
    fetcher = JobFetcher()
    
    print("Testing Job Fetcher...")
    print("-" * 60)
    
    # Test query
    query = "Python Developer"
    location = "San Francisco"
    
    print(f"Searching for: {query} in {location}")
    print()
    
    jobs = fetcher.search_jobs(query, location, max_results=5)
    
    if jobs:
        print(f"Found {len(jobs)} jobs:\n")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['title']}")
            print(f"   Company: {job['company']}")
            print(f"   Location: {job['location']}")
            print(f"   Source: {job['source']}")
            print()
    else:
        print("No jobs found. Please check your API keys in .env file")
