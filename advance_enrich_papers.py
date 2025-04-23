#!/usr/bin/env python3
"""
Script to enrich paper data in Elasticsearch by fetching missing abstracts and PDF URLs
from Semantic Scholar, OpenAlex, and Unpaywall.

Features:
- Runs as a background daemon process
- Implements adaptive rate limiting
- Provides detailed progress tracking and logs
- Handles process state persistence for resumability
- Includes health monitoring and periodic reporting
"""

import argparse
import json
import logging
import os
import sys
import time
import random
import signal
import threading
import queue
from datetime import datetime, timedelta
import concurrent.futures
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union, Set
import atexit

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from tqdm import tqdm
import yaml  # For configuration management

try:
    from pyalex import Works, config as pyalex_config
except ImportError:
    logging.error("pyalex is required. Install with 'pip install pyalex'")
    sys.exit(1)

# --- Configuration Management ---
CONFIG_FILE = "enrich_papers_config.yaml"
DEFAULT_CONFIG = {
    "elasticsearch": {
        "host": "localhost",
        "port": 9200,
        "user": "elastic",
        "password": "",
        "index": "semantic_scholar_papers",
        "timeout": 60,
        "max_retries": 3
    },
    "rate_limits": {
        "semantic_scholar": 3,  # requests per second
        "openalex": 10,
        "unpaywall": 10
    },
    "batch_sizes": {
        "fetch": 100,  # docs per batch from ES
        "update": 500  # docs per bulk update to ES
    },
    "processing": {
        "max_papers": 1000000,
        "threads": 4,  # for concurrent API requests
        "min_citation": 0,
        "max_citation": 1000000
    },
    "retry": {
        "max_retries": 5,
        "base_wait_time": 5  # seconds
    },
    "state_file": "enrich_papers_state.json",
    "log_file": "enrich_papers.log",
    "report_interval": 300,  # seconds between progress reports
    "pyalex_email": "research.tool.user@gmail.com",
    "enable_daemon": False,
    "daemon_sleep": 3600  # seconds to sleep after completing a full run
}

# --- Setup Logging ---
def setup_logging(log_file, log_level=logging.INFO):
    """Configure detailed logging with rotation capability."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates when reloading config
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with info level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler with detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    return root_logger

# --- Email Rotation Manager ---
class EmailRotationManager:
    """Manages email rotation for APIs that need user identification."""
    
    def __init__(self, emails):
        self.emails = emails
        self.current_index = 0
        self.usage_counts = {email: 0 for email in emails}
        self.lock = threading.Lock()
    
    def get_next_email(self):
        """Get the next email in rotation, thread-safe."""
        with self.lock:
            email = self.emails[self.current_index]
            self.usage_counts[email] += 1
            self.current_index = (self.current_index + 1) % len(self.emails)
            return email
    
    def get_stats(self):
        """Return stats about email usage."""
        with self.lock:
            return {
                "total_emails": len(self.emails),
                "usage_counts": dict(self.usage_counts)
            }

# --- Adaptive Rate Limiter ---
class AdaptiveRateLimiter:
    """
    Implements adaptive rate limiting with backoff for API requests.
    Adjusts limits based on response codes and observed latency.
    """
    
    def __init__(self, initial_rates):
        self.rate_limits = initial_rates.copy()
        self.actual_rates = initial_rates.copy()  # Current effective rates
        self.last_request_times = {service: 0 for service in initial_rates}
        self.request_counts = {service: 0 for service in initial_rates}
        self.error_counts = {service: 0 for service in initial_rates}
        self.consecutive_429s = {service: 0 for service in initial_rates}
        self.locks = {service: threading.Lock() for service in initial_rates}
        self.start_time = time.time()
    
    def wait_for_rate_limit(self, service_name):
        """Wait until it's safe to make a request to the specified service."""
        if service_name not in self.rate_limits:
            return  # No limit for this service
        
        with self.locks.get(service_name, threading.Lock()):
            current_time = time.time()
            time_since_last = current_time - self.last_request_times.get(service_name, 0)
            required_interval = 1.0 / self.actual_rates[service_name]
            
            if time_since_last < required_interval:
                sleep_time = required_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_times[service_name] = time.time()
            self.request_counts[service_name] += 1
    
    def record_response(self, service_name, status_code, response_time):
        """Record API response to adjust rate limiting strategy."""
        with self.locks.get(service_name, threading.Lock()):
            # Handle rate limit responses (HTTP 429)
            if status_code == 429:
                self.consecutive_429s[service_name] += 1
                self.error_counts[service_name] += 1
                
                # Reduce rate by 25% after consecutive rate limit errors
                if self.consecutive_429s[service_name] >= 2:
                    self.actual_rates[service_name] = max(
                        1.0,  # Don't go below 1 request per second
                        self.actual_rates[service_name] * 0.75
                    )
                    logging.warning(
                        f"Rate limit hit for {service_name}. Reducing to "
                        f"{self.actual_rates[service_name]:.2f} req/sec"
                    )
            else:
                # Reset consecutive 429s counter on successful response
                self.consecutive_429s[service_name] = 0
                
                # Slowly increase rate if we've been below our target and getting successful responses
                if status_code == 200 and self.actual_rates[service_name] < self.rate_limits[service_name]:
                    # Increase by 5% each successful request, up to the configured limit
                    self.actual_rates[service_name] = min(
                        self.rate_limits[service_name],
                        self.actual_rates[service_name] * 1.05
                    )
    
    def get_stats(self):
        """Return current rate limiting statistics."""
        elapsed = max(1, time.time() - self.start_time)
        return {
            "configured_limits": dict(self.rate_limits),
            "current_limits": dict(self.actual_rates),
            "requests": dict(self.request_counts),
            "errors": dict(self.error_counts),
            "actual_rates": {
                service: count / elapsed
                for service, count in self.request_counts.items()
            }
        }

# --- Process State Management ---
@dataclass
class ProcessState:
    """Represents the current state of the enrichment process for persistence."""
    last_processed_id: Optional[str] = None
    processed_count: int = 0
    updated_count: int = 0
    failed_count: int = 0
    abstract_counts: Dict[str, int] = field(default_factory=lambda: {"semantic_scholar": 0, "openalex": 0, "total": 0})
    pdf_counts: Dict[str, int] = field(default_factory=lambda: {"semantic_scholar": 0, "openalex": 0, "unpaywall": 0, "total": 0})
    error_counts: Dict[str, int] = field(default_factory=lambda: {"semantic_scholar": 0, "openalex": 0, "unpaywall": 0, "es_update": 0})
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    already_processed_ids: Set[str] = field(default_factory=set)
    citation_range: Dict[str, int] = field(default_factory=lambda: {"min": 0, "max": 0})
    
    def to_dict(self):
        """Convert state to serializable dictionary."""
        result = asdict(self)
        # Convert set to list for JSON serialization
        result["already_processed_ids"] = list(self.already_processed_ids)
        return result
    
    @classmethod
    def from_dict(cls, data):
        """Create state object from dictionary."""
        if not data:
            return cls()
        
        # Convert list back to set
        if "already_processed_ids" in data:
            data["already_processed_ids"] = set(data["already_processed_ids"])
        
        return cls(**data)

class StateManager:
    """Manages process state persistence for recovery."""
    
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = self.load_state()
        self.lock = threading.Lock()
        # Register save on exit
        atexit.register(self.save_state)
    
    def load_state(self):
        """Load process state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    logging.info(f"Loaded previous state: processed {data.get('processed_count', 0)} papers")
                    return ProcessState.from_dict(data)
        except Exception as e:
            logging.warning(f"Error loading state file: {e}")
        
        return ProcessState()
    
    def save_state(self):
        """Save current process state to file."""
        with self.lock:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state.to_dict(), f)
                logging.debug("State saved successfully")
            except Exception as e:
                logging.error(f"Error saving state: {e}")
    
    def record_progress(self, paper_id=None, **kwargs):
        """Update state with progress information."""
        with self.lock:
            if paper_id:
                self.state.last_processed_id = paper_id
                self.state.already_processed_ids.add(paper_id)
            
            # Update any provided fields
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    current_val = getattr(self.state, key)
                    if isinstance(current_val, dict) and isinstance(value, dict):
                        # Update nested dictionary
                        for subkey, subvalue in value.items():
                            if subkey in current_val:
                                current_val[subkey] += subvalue
                    else:
                        # Update simple field
                        setattr(self.state, key, current_val + value)
            
            self.state.last_update_time = time.time()
            
            # Periodically save state (every 100 papers)
            if self.state.processed_count % 100 == 0:
                self.save_state()
    
    def should_process_paper(self, paper_id):
        """Check if paper should be processed or skipped (already done)."""
        return paper_id not in self.state.already_processed_ids
    
    def get_stats(self):
        """Return current progress statistics."""
        with self.lock:
            elapsed = max(1, time.time() - self.state.start_time)
            papers_per_sec = self.state.processed_count / elapsed
            
            # Only calculate if any papers were updated
            update_success_rate = (
                (self.state.updated_count / max(1, self.state.updated_count + self.state.failed_count)) * 100
                if (self.state.updated_count + self.state.failed_count) > 0
                else 0
            )
            
            return {
                "papers_processed": self.state.processed_count,
                "papers_updated": self.state.updated_count,
                "papers_failed": self.state.failed_count,
                "abstracts_found": dict(self.state.abstract_counts),
                "pdfs_found": dict(self.state.pdf_counts),
                "errors": dict(self.state.error_counts),
                "elapsed_time": elapsed,
                "papers_per_second": papers_per_sec,
                "update_success_rate": update_success_rate,
                "last_paper_id": self.state.last_processed_id
            }

# --- API Request Handler ---
class APIRequestHandler:
    """Handles API requests with retries, rate limiting and error tracking."""
    
    def __init__(self, rate_limiter, email_manager=None, retry_config=None):
        self.rate_limiter = rate_limiter
        self.email_manager = email_manager
        self.retry_config = retry_config or {"max_retries": 5, "base_wait_time": 5}
        self.session = requests.Session()
        # Add sensible default timeouts and headers
        self.session.headers.update({
            "User-Agent": "EnrichPapers/1.0 (Academic Research Tool)"
        })
    
    def make_request(self, service_name, method, url, **kwargs):
        """Make API request with retry logic and rate limiting."""
        max_retries = self.retry_config["max_retries"]
        base_wait_time = self.retry_config["base_wait_time"]
        retry_count = 0
        start_time = time.time()
        
        # Ensure timeout is set
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60
        
        while retry_count < max_retries:
            try:
                # Respect rate limits before making request
                self.rate_limiter.wait_for_rate_limit(service_name)
                
                # Make actual request
                logging.debug(f"Making {method.upper()} request to {service_name}: {url}")
                response = self.session.request(method, url, **kwargs)
                
                # Record response for rate limiter to adapt
                response_time = time.time() - start_time
                self.rate_limiter.record_response(service_name, response.status_code, response_time)
                
                # Handle response based on status code
                if response.status_code == 200:
                    return response
                
                # Rate limiting - use Retry-After header if available
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", base_wait_time * (2 ** retry_count)))
                    logging.warning(f"{service_name} rate limit hit (429). Retrying after {retry_after} seconds...")
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                
                # Server errors - use exponential backoff
                elif response.status_code >= 500:
                    backoff = base_wait_time * (2 ** retry_count)
                    logging.warning(f"{service_name} server error ({response.status_code}). Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                    retry_count += 1
                    continue
                
                # Handle client errors
                elif response.status_code >= 400:
                    if response.status_code == 404:
                        logging.debug(f"{service_name}: Resource not found (404): {url}")
                    else:
                        logging.warning(f"{service_name} client error ({response.status_code}): {url} - {response.text[:200]}")
                    
                    return response
            
            except requests.exceptions.Timeout:
                backoff = base_wait_time * (2 ** retry_count)
                logging.warning(f"{service_name} request timed out. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retry_count += 1
            
            except requests.exceptions.ConnectionError:
                backoff = base_wait_time * (2 ** retry_count)
                logging.warning(f"{service_name} connection error. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retry_count += 1
            
            except Exception as e:
                backoff = base_wait_time * (2 ** retry_count)
                logging.error(f"Unexpected error during {service_name} request: {e}", exc_info=True)
                time.sleep(backoff)
                retry_count += 1
        
        logging.error(f"Failed {service_name} request to {url} after {max_retries} retries.")
        return None

# --- Elasticsearch Client ---
class ESClient:
    """Handles Elasticsearch operations with proper error handling."""
    
    def __init__(self, host, port, user, password, timeout=60, max_retries=3):
        self.connection_params = {
            "hosts": [f"http://{host}:{port}"],
            "http_auth": (user, password),
            "timeout": timeout,
            "max_retries": max_retries,
            "retry_on_timeout": True
        }
        self.client = None
        self.connect()
    
    def connect(self):
        """Establish connection to Elasticsearch."""
        try:
            self.client = Elasticsearch(**self.connection_params)
            if self.client.ping():
                logging.info("Successfully connected to Elasticsearch")
                return True
            else:
                logging.error("Failed to connect to Elasticsearch (ping failed)")
                return False
        except Exception as e:
            logging.error(f"Error connecting to Elasticsearch: {e}")
            return False
    
    def get_papers_to_enrich(self, index_name, min_citation, max_citation, batch_size=100, last_id=None):
        """
        Fetch papers that need enrichment using the scan helper with resume capability.
        """
        # Build the query
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "citationCount": {
                                    "gte": min_citation,
                                    "lte": max_citation
                                }
                            }
                        }
                    ],
                    "filter": [
                        {
                            "bool": {
                                "should": [
                                    { "bool": { "must_not": { "exists": { "field": "abstract" } } } },
                                    { "term": { "abstract": "" } },
                                    { "bool": { "must_not": { "exists": { "field": "openAccessPdf.url" } } } }
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    ]
                }
            },
            "_source": ["paperId", "title", "externalIds", "abstract", "openAccessPdf"],
            "sort": [{"_id": "asc"}]  # Sort by _id for reliable pagination
        }
        
        # Add search_after for resuming from last processed ID
        if last_id:
            query["search_after"] = [last_id]
            logging.info(f"Resuming scan from document ID: {last_id}")
        
        logging.info(
            f"Fetching papers from index '{index_name}' with citation range "
            f"[{min_citation}, {max_citation}] missing abstract or PDF URL..."
        )
        
        try:
            scan_generator = scan(
                client=self.client,
                index=index_name,
                query=query,
                size=batch_size,
                scroll='5m',
                preserve_order=True  # Important for resuming correctly
            )
            return scan_generator
        except Exception as e:
            logging.error(f"Error fetching papers using scan: {e}")
            return None
    
    def bulk_update(self, index_name, actions):
        """Perform bulk update to Elasticsearch with error handling."""
        if not actions:
            return 0, 0  # No actions to process
        
        try:
            success, errors = bulk(
                self.client, 
                actions, 
                raise_on_error=False,
                refresh=False,  # Set to True only if you need immediate visibility
                chunk_size=len(actions)  # Keep the batch as a single chunk
            )
            return success, len(errors) if errors else 0
        except Exception as e:
            logging.error(f"Error performing bulk update: {e}")
            return 0, len(actions)  # Consider all failed

# --- Data Enrichment Services ---
class EnrichmentService:
    """Base class for enrichment services."""
    
    def __init__(self, api_handler):
        self.api_handler = api_handler
    
    def enrich(self, paper_data):
        """
        Enrich paper data with additional information.
        To be implemented by subclasses.
        """
        raise NotImplementedError

class SemanticScholarEnricher(EnrichmentService):
    """Enriches papers using Semantic Scholar API."""
    
    def enrich(self, paper_data):
        """Fetch details from Semantic Scholar Single Paper API."""
        paper_id = paper_data.get('paperId')
        
        if not paper_id:
            return None
        
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract,isOpenAccess,openAccessPdf"
        
        headers = {}  # Add API key if available
        response = self.api_handler.make_request('semantic_scholar', 'get', url, headers=headers)
        
        if not response or response.status_code != 200:
            return None
        
        try:
            data = response.json()
            update_data = {}
            
            # Extract abstract if available
            if data.get('abstract'):
                update_data['abstract'] = data['abstract']
            
            # Check for PDF URL
            s2_pdf_data = data.get('openAccessPdf')
            if data.get('isOpenAccess') and s2_pdf_data and s2_pdf_data.get('url'):
                s2_pdf_url = s2_pdf_data['url']
                if s2_pdf_url and isinstance(s2_pdf_url, str) and s2_pdf_url.lower().endswith('.pdf'):
                    update_data['openAccessPdf'] = {
                        'url': s2_pdf_url,
                        'status': s2_pdf_data.get('status', 'oa_found_s2')
                    }
            
            return update_data if update_data else None
        
        except json.JSONDecodeError:
            logging.warning(f"Failed to decode JSON from Semantic Scholar for {paper_id}")
            return None

class OpenAlexEnricher(EnrichmentService):
    """Enriches papers using OpenAlex API."""
    
    def __init__(self, api_handler, email):
        super().__init__(api_handler)
        pyalex_config.email = email
    
    def reconstruct_abstract(self, inverted_index):
        """Reconstruct abstract from OpenAlex inverted index."""
        if not inverted_index:
            return None
        try:
            word_positions = [(word, pos) for word, positions in inverted_index.items() for pos in positions]
            sorted_words = sorted(word_positions, key=lambda x: int(x[1]))
            return " ".join(word for word, pos in sorted_words)
        except Exception as e:
            logging.error(f"Error reconstructing abstract: {e}")
            return None
    
    def enrich(self, paper_data):
        """Fetch details from OpenAlex using DOI or title."""
        doi = paper_data.get('externalIds', {}).get('DOI')
        title = paper_data.get('title')
        
        if not doi and not title:
            return None
        
        try:
            # Apply rate limit manually before using pyalex
            self.api_handler.rate_limiter.wait_for_rate_limit('openalex')
            
            work = None
            if doi:
                doi_cleaned = doi.replace("https://doi.org/", "")
                results = Works().filter(doi=doi_cleaned).get()
                if results:
                    work = results[0]
            elif title:
                results = Works().filter(title={'search': title}).get()
                if results:
                    work = results[0]
            
            if not work:
                return None
            
            # Record successful API call
            self.api_handler.rate_limiter.record_response('openalex', 200, 0)
            
            update_data = {}
            
            # Extract abstract
            abstract_inverted = work.get("abstract_inverted_index")
            if abstract_inverted:
                abstract_text = self.reconstruct_abstract(abstract_inverted)
                if abstract_text:
                    update_data['abstract'] = abstract_text
            
            # Extract PDF URL
            best_oa_loc = work.get("best_oa_location")
            if isinstance(best_oa_loc, dict):
                oa_pdf_url = best_oa_loc.get("pdf_url")
                if oa_pdf_url and isinstance(oa_pdf_url, str) and oa_pdf_url.lower().endswith('.pdf'):
                    update_data['openAccessPdf'] = {
                        'url': oa_pdf_url,
                        'status': best_oa_loc.get("license") or best_oa_loc.get('version') or 'oa_found_openalex'
                    }
            
            return update_data if update_data else None
        
        except Exception as e:
            # Record failed API call
            self.api_handler.rate_limiter.record_response('openalex', 500, 0)
            logging.error(f"Error querying OpenAlex: {e}")
            return None

class UnpaywallEnricher(EnrichmentService):
    """Enriches papers using Unpaywall API."""
    
    def __init__(self, api_handler, email_manager):
        super().__init__(api_handler)
        self.email_manager = email_manager
    
    def enrich(self, paper_data):
        """Fetch open access details from Unpaywall API."""
        doi = paper_data.get('externalIds', {}).get('DOI')
        
        if not doi:
            return None
        
        doi_cleaned = doi.replace("https://doi.org/", "")
        selected_email = self.email_manager.get_next_email()
        
        url = f"https://api.unpaywall.org/v2/{doi_cleaned}?email={selected_email}"
        response = self.api_handler.make_request('unpaywall', 'get', url)
        
        if not response or response.status_code != 200:
            return None
        
        try:
            data = response.json()
            oa_location = data.get("best_oa_location")
            
            if oa_location and oa_location.get("url_for_pdf"):
                upw_pdf_url = oa_location.get("url_for_pdf")
                
                if upw_pdf_url and isinstance(upw_pdf_url, str) and upw_pdf_url.lower().endswith('.pdf'):
                    return {
                        'openAccessPdf': {
                            'url': upw_pdf_url,
                            'status': oa_location.get("version") or oa_location.get("license") or 'oa_found_unpaywall'
                        }
                    }
            
            return None
        
        except json.JSONDecodeError:
            logging.warning(f"Failed to decode JSON from Unpaywall for {doi}")
            return None

# --- Progress Reporter ---
class ProgressReporter:
    """Handles periodic reporting of enrichment progress."""
    
    def __init__(self, state_manager, rate_limiter, email_manager=None, report_interval=300):
        self.state_manager = state_manager
        self.rate_limiter = rate_limiter
        self.email_manager = email_manager
        self.report_interval = report_interval
        self.stop_event = threading.Event()
        self.thread = None
    
    def start(self):
        """Start the progress reporter thread."""
        self.thread = threading.Thread(target=self._reporting_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the progress reporter thread."""
        if self.thread:
            self.stop_event.set()
            self.thread.join(timeout=1.0)
    
    def _reporting_loop(self):
        """Background thread that periodically reports progress."""
        while not self.stop_event.is_set():
            self._generate_report()
            # Wait for next report interval or until stopped
            self.stop_event.wait(self.report_interval)
    
    def _generate_report(self):
        """Generate and log a progress report."""
        stats = self.state_manager.get_stats()
        rate_stats = self.rate_limiter.get_stats()
        
        # Format elapsed time
        elapsed_seconds = stats["elapsed_time"]
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_formatted = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Create report
        report = [
            "="*50,
            "ENRICHMENT PROGRESS REPORT",
            "="*50,
            f"Elapsed time: {elapsed_formatted}",
            f"Papers processed: {stats['papers_processed']:,}",
            f"Processing rate: {stats['papers_per_second']:.2f} papers/sec",
            f"Updates successful: {stats['papers_updated']:,}",
            f"Updates failed: {stats['papers_failed']:,}",
            f"Success rate: {stats['update_success_rate']:.1f}%",
            "-"*50,
            "Content Found:",
            f"  Abstracts: {stats['abstracts_found']['total']:,} total",
            f"    - Semantic Scholar: {stats['abstracts_found']['semantic_scholar']:,}",
            f"    - OpenAlex: {stats['abstracts_found']['openalex']:,}",
            f"  PDF URLs: {stats['pdfs_found']['total']:,} total",
            f"    - Semantic Scholar: {stats['pdfs_found']['semantic_scholar']:,}",
            f"    - OpenAlex: {stats['pdfs_found']['openalex']:,}",
            f"    - Unpaywall: {stats['pdfs_found']['unpaywall']:,}",
            "-"*50,
            "API Statistics:",
            f"  Requests (total): {sum(rate_stats['requests'].values()):,}",
            f"  Errors (total): {sum(rate_stats['errors'].values()):,}",
            f"  Current rate limits: {', '.join(f'{svc}: {rate:.1f}/s' for svc, rate in rate_stats['current_limits'].items())}",
            "="*50
        ]
        
        # Log the report
        report_text = "\n".join(report)
        logging.info(f"\n{report_text}")
        
        # Also print to console directly for visibility
        print(f"\n{report_text}")


# --- Enrichment Coordinator ---
class EnrichmentCoordinator:
    """
    Coordinates the entire enrichment process, including:
    - Paper fetching from Elasticsearch
    - Distributing enrichment tasks to worker threads
    - Processing results and updating Elasticsearch
    - Tracking progress and generating reports
    """
    
    def __init__(self, config):
        self.config = config
        self.stop_event = threading.Event()
        self.update_queue = queue.Queue(maxsize=1000)  # Buffer for ES updates
        
        # Initialize components
        self.es_client = ESClient(
            config["elasticsearch"]["host"],
            config["elasticsearch"]["port"],
            config["elasticsearch"]["user"],
            config["elasticsearch"]["password"],
            timeout=config["elasticsearch"]["timeout"],
            max_retries=config["elasticsearch"]["max_retries"]
        )
        
        self.rate_limiter = AdaptiveRateLimiter(config["rate_limits"])
        
        # List of unpaywall emails - add your emails here
        UNPAYWALL_EMAILS = [
            "research.tool.user@gmail.com",
            "academic.query@gmail.com",
            "openaccess.finder@gmail.com",
            "pdf.retriever@gmail.com",
            "scholarly.api@gmail.com"
        ]
        
        self.email_manager = EmailRotationManager(UNPAYWALL_EMAILS)
        
        self.state_manager = StateManager(config["state_file"])
        
        self.api_handler = APIRequestHandler(
            self.rate_limiter,
            self.email_manager,
            config["retry"]
        )
        
        # Initialize enrichers
        self.enrichers = {
            "semantic_scholar": SemanticScholarEnricher(self.api_handler),
            "openalex": OpenAlexEnricher(self.api_handler, config["pyalex_email"]),
            "unpaywall": UnpaywallEnricher(self.api_handler, self.email_manager)
        }
        
        # Initialize reporter
        self.reporter = ProgressReporter(
            self.state_manager,
            self.rate_limiter,
            self.email_manager,
            config["report_interval"]
        )
        
        # Thread pool for API requests
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config["processing"]["threads"],
            thread_name_prefix="enricher"
        )
        
        # Update worker thread
        self.update_thread = None
    
    def enrich_paper(self, paper_doc):
        """Process a single paper document for enrichment."""
        paper_id = paper_doc["_id"]
        source = paper_doc["_source"]
        
        # Skip if already processed
        if not self.state_manager.should_process_paper(paper_id):
            return None
        
        needs_update = False
        update_doc = {}
        
        # Determine what's missing
        missing_abstract = not source.get('abstract')
        pdf_info = source.get('openAccessPdf', {})
        missing_pdf_url = not isinstance(pdf_info, dict) or not pdf_info.get('url')
        
        # Extract identifiers
        paper_id_s2 = source.get('paperId')
        doi = source.get('externalIds', {}).get('DOI')
        title = source.get('title')
        
        # 1. Try Semantic Scholar
        if paper_id_s2 and (missing_abstract or missing_pdf_url):
            try:
                s2_data = self.enrichers["semantic_scholar"].enrich(source)
                if s2_data:
                    if missing_abstract and 'abstract' in s2_data:
                        update_doc['abstract'] = s2_data['abstract']
                        self.state_manager.record_progress(
                            abstract_counts={"semantic_scholar": 1, "total": 1}
                        )
                        missing_abstract = False
                        needs_update = True
                    
                    if missing_pdf_url and 'openAccessPdf' in s2_data:
                        update_doc['openAccessPdf'] = s2_data['openAccessPdf']
                        self.state_manager.record_progress(
                            pdf_counts={"semantic_scholar": 1, "total": 1}
                        )
                        missing_pdf_url = False
                        needs_update = True
            except Exception as e:
                logging.error(f"Error enriching from Semantic Scholar: {e}")
                self.state_manager.record_progress(error_counts={"semantic_scholar": 1})
        
        # 2. Try OpenAlex if still missing data
        if (missing_abstract or missing_pdf_url) and (doi or title):
            try:
                oa_data = self.enrichers["openalex"].enrich(source)
                if oa_data:
                    if missing_abstract and 'abstract' in oa_data:
                        update_doc['abstract'] = oa_data['abstract']
                        self.state_manager.record_progress(
                            abstract_counts={"openalex": 1, "total": 1}
                        )
                        missing_abstract = False
                        needs_update = True
                    
                    if missing_pdf_url and 'openAccessPdf' in oa_data:
                        # Only update if S2 didn't already find one
                        if 'openAccessPdf' not in update_doc:
                            update_doc['openAccessPdf'] = oa_data['openAccessPdf']
                            self.state_manager.record_progress(
                                pdf_counts={"openalex": 1, "total": 1}
                            )
                            missing_pdf_url = False
                            needs_update = True
            except Exception as e:
                logging.error(f"Error enriching from OpenAlex: {e}")
                self.state_manager.record_progress(error_counts={"openalex": 1})
        
        # 3. Try Unpaywall if still missing PDF URL
        if missing_pdf_url and doi:
            try:
                upw_data = self.enrichers["unpaywall"].enrich(source)
                if upw_data and 'openAccessPdf' in upw_data:
                    # Only update if not already found
                    if 'openAccessPdf' not in update_doc:
                        update_doc['openAccessPdf'] = upw_data['openAccessPdf']
                        self.state_manager.record_progress(
                            pdf_counts={"unpaywall": 1, "total": 1}
                        )
                        missing_pdf_url = False
                        needs_update = True
            except Exception as e:
                logging.error(f"Error enriching from Unpaywall: {e}")
                self.state_manager.record_progress(error_counts={"unpaywall": 1})
        
        # Add to update queue if changes were made
        if needs_update and update_doc:
            update_action = {
                "_op_type": "update",
                "_index": self._get_index_name(),
                "_id": paper_id,
                "doc": update_doc
            }
            
            # Put in queue for bulk processing
            self.update_queue.put(update_action)
            self.state_manager.record_progress(paper_id=paper_id, processed_count=1)
            return update_action
        
        # No updates needed, just record progress
        self.state_manager.record_progress(paper_id=paper_id, processed_count=1)
        return None
    
    def _get_index_name(self):
        return self.config["elasticsearch"]["index"]
    
    def _update_worker(self):
        """Background thread that processes queued updates in batches."""
        batch_size = self.config["batch_sizes"]["update"]
        batch = []
        last_flush_time = time.time()
        flush_interval = 30  # Flush every 30 seconds even if batch not full
        
        while not self.stop_event.is_set() or not self.update_queue.empty():
            try:
                # Try to get an item with timeout
                try:
                    action = self.update_queue.get(timeout=1.0)
                    batch.append(action)
                    self.update_queue.task_done()
                except queue.Empty:
                    # No items in queue, check if we should flush anyway
                    if batch and time.time() - last_flush_time > flush_interval:
                        pass  # Will flush below
                    else:
                        continue  # Try again
                
                # Process batch if full or time interval exceeded
                if len(batch) >= batch_size or (batch and time.time() - last_flush_time > flush_interval):
                    success_count, failed_count = self.es_client.bulk_update(
                        self._get_index_name(), batch
                    )
                    
                    self.state_manager.record_progress(
                        updated_count=success_count,
                        failed_count=failed_count
                    )
                    
                    if failed_count > 0:
                        self.state_manager.record_progress(error_counts={"es_update": failed_count})
                    
                    batch = []
                    last_flush_time = time.time()
            
            except Exception as e:
                logging.error(f"Error in update worker: {e}", exc_info=True)
                # Don't lose the batch if possible
                if batch:
                    try:
                        success_count, failed_count = self.es_client.bulk_update(
                            self._get_index_name(), batch
                        )
                        self.state_manager.record_progress(
                            updated_count=success_count,
                            failed_count=failed_count
                        )
                    except Exception:
                        # If that also fails, increment error count
                        self.state_manager.record_progress(
                            failed_count=len(batch),
                            error_counts={"es_update": len(batch)}
                        )
                    batch = []
                time.sleep(5)  # Brief pause after error
        
        # Flush any remaining items
        if batch:
            try:
                success_count, failed_count = self.es_client.bulk_update(
                    self._get_index_name(), batch
                )
                self.state_manager.record_progress(
                    updated_count=success_count,
                    failed_count=failed_count
                )
            except Exception as e:
                logging.error(f"Error in final batch update: {e}")
                self.state_manager.record_progress(
                    failed_count=len(batch),
                    error_counts={"es_update": len(batch)}
                )
    
    def start(self):
        """Start the enrichment process."""
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Start reporter
        self.reporter.start()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_worker,
            daemon=True,
            name="es_updater"
        )
        self.update_thread.start()
        
        # Begin processing
        self._run_enrichment()
    
    def stop(self):
        """Stop the enrichment process gracefully."""
        logging.info("Stopping enrichment process...")
        self.stop_event.set()
        
        # Stop reporter
        self.reporter.stop()
        
        # Wait for threads to finish
        if self.update_thread and self.update_thread.is_alive():
            logging.info("Waiting for update thread to complete...")
            self.update_thread.join(timeout=60.0)
        
        # Wait for all updates to complete
        logging.info("Waiting for update queue to empty...")
        try:
            self.update_queue.join(timeout=60.0)
        except Exception:
            logging.warning("Timed out waiting for update queue to empty")
        
        # Shutdown thread pool
        logging.info("Shutting down thread pool...")
        self.thread_pool.shutdown(wait=True, cancel_futures=False)
        
        # Save final state
        self.state_manager.save_state()
        
        # Generate final report
        self.reporter._generate_report()
        
        logging.info("Enrichment process stopped")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logging.info(f"Received signal {sig}, shutting down...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _run_enrichment(self):
        """Main enrichment loop."""
        # Get configuration
        index_name = self.config["elasticsearch"]["index"]
        min_citation = self.config["processing"]["min_citation"]
        max_citation = self.config["processing"]["max_citation"]
        batch_size = self.config["batch_sizes"]["fetch"]
        max_papers = self.config["processing"]["max_papers"]
        
        # Update state with citation range
        self.state_manager.state.citation_range = {"min": min_citation, "max": max_citation}
        
        # Get papers from Elasticsearch
        last_id = self.state_manager.state.last_processed_id
        papers_generator = self.es_client.get_papers_to_enrich(
            index_name, min_citation, max_citation, batch_size, last_id
        )
        
        if not papers_generator:
            logging.error("Failed to get papers from Elasticsearch")
            return
        
        # Process papers in thread pool
        future_to_paper = {}
        papers_processed = 0
        
        try:
            # Main processing loop
            for doc in papers_generator:
                if self.stop_event.is_set():
                    logging.info("Stop event detected, halting paper processing")
                    break
                
                if papers_processed >= max_papers:
                    logging.info(f"Reached max papers limit ({max_papers})")
                    break
                
                # Skip if already processed (double-check)
                if not self.state_manager.should_process_paper(doc["_id"]):
                    logging.debug(f"Skipping already processed paper {doc['_id']}")
                    continue
                
                # Submit to thread pool
                future = self.thread_pool.submit(self.enrich_paper, doc)
                future_to_paper[future] = doc["_id"]
                
                # Process completed futures to avoid memory buildup
                completed = [f for f in future_to_paper if f.done()]
                for future in completed:
                    paper_id = future_to_paper[future]
                    try:
                        # Just to catch any exceptions
                        future.result()
                    except Exception as e:
                        logging.error(f"Error processing paper {paper_id}: {e}")
                    del future_to_paper[future]
                
                papers_processed += 1
                
                # Periodically generate progress report
                if papers_processed % 5000 == 0:
                    self.reporter._generate_report()
            
            # Wait for remaining tasks
            for future in concurrent.futures.as_completed(future_to_paper):
                paper_id = future_to_paper[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing paper {paper_id}: {e}")
        
        except Exception as e:
            logging.error(f"Error in enrichment process: {e}", exc_info=True)
        
        finally:
            # Wait for tasks to complete before returning
            logging.info("Waiting for remaining tasks to complete...")
            for future in concurrent.futures.as_completed(list(future_to_paper.keys())):
                try:
                    future.result()
                except Exception:
                    pass
    
    def run_daemon(self):
        """Run in daemon mode, periodically processing papers."""
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                logging.info("Starting enrichment cycle")
                self._run_enrichment()
                logging.info("Enrichment cycle completed")
            except Exception as e:
                logging.error(f"Error in enrichment cycle: {e}", exc_info=True)
            
            # Sleep until next cycle if daemon mode enabled
            if self.config["enable_daemon"] and not self.stop_event.is_set():
                sleep_time = self.config["daemon_sleep"]
                elapsed = time.time() - start_time
                
                if elapsed < sleep_time:
                    remaining = sleep_time - elapsed
                    logging.info(f"Sleeping for {remaining:.1f} seconds until next cycle")
                    
                    # Wait for stop event or sleep time
                    self.stop_event.wait(remaining)
            else:
                # Not in daemon mode, exit after one cycle
                break

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Enrich Elasticsearch paper data with abstracts and PDF URLs.")
    
    # Configuration options
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true", help="Create a default configuration file")
    
    # Elasticsearch options
    parser.add_argument("--host", help="Elasticsearch host")
    parser.add_argument("--port", type=int, help="Elasticsearch port")
    parser.add_argument("--user", help="Elasticsearch user")
    parser.add_argument("--password", help="Elasticsearch password")
    parser.add_argument("--index", help="Elasticsearch index name")
    
    # Processing options
    parser.add_argument("--min-citation", type=int, help="Minimum citation count")
    parser.add_argument("--max-citation", type=int, help="Maximum citation count")
    parser.add_argument("--fetch-batch-size", type=int, help="Number of papers to fetch per batch")
    parser.add_argument("--update-batch-size", type=int, help="Number of updates to send in bulk")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to process")
    parser.add_argument("--threads", type=int, help="Number of concurrent worker threads")
    
    # Runtime options
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode, processing periodically")
    parser.add_argument("--daemon-sleep", type=int, help="Seconds to sleep between daemon runs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--state-file", help="File to store process state")
    parser.add_argument("--report-interval", type=int, help="Seconds between progress reports")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed paper")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        try:
            with open(CONFIG_FILE, 'w') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            print(f"Default configuration created at {CONFIG_FILE}")
            print("Please edit this file with your settings and run again.")
            return
        except Exception as e:
            print(f"Error creating config file: {e}")
            return
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f)
                # Deep update the configuration
                for section, values in file_config.items():
                    if section in config and isinstance(config[section], dict) and isinstance(values, dict):
                        config[section].update(values)
                    else:
                        config[section] = values
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Override config with command line arguments
    if args.host:
        config["elasticsearch"]["host"] = args.host
    if args.port:
        config["elasticsearch"]["port"] = args.port
    if args.user:
        config["elasticsearch"]["user"] = args.user
    if args.password:
        config["elasticsearch"]["password"] = args.password
    if args.index:
        config["elasticsearch"]["index"] = args.index
    
    if args.min_citation is not None:
        config["processing"]["min_citation"] = args.min_citation
    if args.max_citation is not None:
        config["processing"]["max_citation"] = args.max_citation
    if args.fetch_batch_size:
        config["batch_sizes"]["fetch"] = args.fetch_batch_size
    if args.update_batch_size:
        config["batch_sizes"]["update"] = args.update_batch_size
    if args.max_papers:
        config["processing"]["max_papers"] = args.max_papers
    if args.threads:
        config["processing"]["threads"] = args.threads
    
    if args.daemon:
        config["enable_daemon"] = True
    if args.daemon_sleep:
        config["daemon_sleep"] = args.daemon_sleep
    if args.state_file:
        config["state_file"] = args.state_file
    if args.report_interval:
        config["report_interval"] = args.report_interval
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(config["log_file"], log_level)
    
    # Log configuration
    logger.info(f"Starting enrichment process with the following configuration:")
    for section, values in config.items():
        if isinstance(values, dict):
            logger.info(f"{section}:")
            for key, value in values.items():
                # Don't log passwords
                if 'password' in key.lower():
                    logger.info(f"  {key}: ********")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.info(f"{section}: {values}")
    
    # Start the enrichment process
    try:
        coordinator = EnrichmentCoordinator(config)
        
        # Register cleanup on exit
        def cleanup():
            try:
                coordinator.stop()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        atexit.register(cleanup)
        
        # Run either in daemon mode or one-time mode
        if config["enable_daemon"]:
            logger.info("Running in daemon mode")
            coordinator.run_daemon()
        else:
            logger.info("Running in one-time mode")
            coordinator.start()
            coordinator.stop()
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("Enrichment process completed")

# --- Startup Script ---
if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("This script requires Python 3.7 or higher")
        sys.exit(1)
    
    print("Starting paper enrichment script...")
    
    try:
        print("Initializing main function")
        main()
        print("Main function completed")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
