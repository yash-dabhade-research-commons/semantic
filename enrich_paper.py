#!/usr/bin/env python3
"""
Script to enrich paper data in Elasticsearch by fetching missing abstracts and PDF URLs
from Semantic Scholar, OpenAlex, and Unpaywall.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
import random # Import random module

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from tqdm import tqdm
from pyalex import Works, config as pyalex_config # Reusing pyalex

# --- Configuration ---
# Configure pyalex - Reuse email from pyalex_script
pyalex_config.email = "joseph.m.garcia@gmail.com"
# UNPAYWALL_EMAIL = "joseph.m.garcia@gmail.com" # Email for Unpaywall API

# List of emails for Unpaywall API rotation
UNPAYWALL_EMAILS = [
    "researcher1@gmail.com",
    "student.query@gmail.com",
    "data.fetcher@gmail.com",
    "api.user.1@gmail.com",
    "unpaywall.access@gmail.com",
    "literature.search@gmail.com",
    "academic.query@gmail.com",
    "openaccess.finder@gmail.com",
    "pdf.retriever@gmail.com",
    "scholarly.api@gmail.com",
    "research.tool.user@gmail.com",
    "library.access@gmail.com",
    "data.miner@gmail.com",
    "api.client.beta@gmail.com",
    "oa.seeker@gmail.com",
    "study.group@gmail.com",
    "lab.assistant@gmail.com",
    "access.point@gmail.com",
    "doc.finder@gmail.com",
    "api.tester@gmail.com",
    "another.researcher@gmail.com",
    "postgrad.query@gmail.com",
    "script.user@gmail.com",
    "api.user.2@gmail.com",
    "unpaywall.client@gmail.com",
    "search.agent@gmail.com",
    "project.lead@gmail.com",
    "finder.bot@gmail.com",
    "url.checker@gmail.com",
    "dev.access@gmail.com",
    "research.team@gmail.com",
    "grad.student@gmail.com",
    "automated.fetch@gmail.com",
    "api.user.3@gmail.com",
    "open.access.query@gmail.com",
    "uk.research@gmail.com",
    "data.analysis@gmail.com",
    "link.finder@gmail.com",
    "pdf.link.checker@gmail.com",
    "service.account@gmail.com",
    "librarian.query@gmail.com",
    "faculty.member@gmail.com",
    "network.tool@gmail.com",
    "api.user.4@gmail.com",
    "unpaywall.requester@gmail.com",
    "dept.search@gmail.com",
    "analysis.script@gmail.com",
    "paper.finder@gmail.com",
    "link.checker.script@gmail.com",
    "app.backend@gmail.com",
    # Add more emails here to reach 50-80 if needed
    "research51@gmail.com",
    "student52@gmail.com",
    "data53@gmail.com",
    "apiuser54@gmail.com",
    "unpaywall55@gmail.com",
    "search56@gmail.com",
    "query57@gmail.com",
    "access58@gmail.com",
    "pdf59@gmail.com",
    "scholarly60@gmail.com"
]

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enrich_papers.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("paper_enricher")

# --- Rate Limiting & Retry Logic (Adapted from bulk_search.py) ---
# Global state for rate limiting (simple approach)
last_request_times = {
    'semantic_scholar': 0,
    'openalex': 0,
    'unpaywall': 0
}
# Define rate limits (requests per second)
# Semantic Scholar (non-bulk): ~3/sec recommended? Check docs. Assume 3.
# OpenAlex Polite Pool: 10/sec
# Unpaywall: 10/sec recommended
rate_limits = {
    'semantic_scholar': 1,
    'openalex': 10,
    'unpaywall': 10
}

def _respect_rate_limit(service_name):
    """Ensure we don't exceed the rate limit for a specific service."""
    if service_name not in rate_limits:
        return # No limit for this service

    current_time = time.time()
    time_since_last = current_time - last_request_times.get(service_name, 0)
    limit = rate_limits[service_name]
    required_interval = 1.0 / limit

    if time_since_last < required_interval:
        sleep_time = required_interval - time_since_last
        # logger.debug(f"Rate limit for {service_name}: Sleeping for {sleep_time:.3f}s")
        time.sleep(sleep_time)

    last_request_times[service_name] = time.time()

def make_request_with_retry(service_name, method, url, **kwargs):
    """Make an HTTP request with retry logic, respecting rate limits."""
    max_retries = 5
    retry_count = 0
    base_wait_time = 5 # seconds

    while retry_count < max_retries:
        try:
            _respect_rate_limit(service_name)

            # Add timeout to requests
            if 'timeout' not in kwargs:
                kwargs['timeout'] = 60 # 60 seconds timeout

            logger.debug(f"Making {method.upper()} request to {service_name}: {url}")
            response = requests.request(method, url, **kwargs)

            # Success
            if response.status_code == 200:
                # logger.debug(f"{service_name} request successful (status {response.status_code})")
                return response

            # Rate limiting
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", base_wait_time * (retry_count + 1)))
                logger.warning(f"{service_name} rate limit hit (429). Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                retry_count += 1
                continue

            # Server errors
            elif response.status_code >= 500:
                 logger.warning(f"{service_name} server error ({response.status_code}). Retrying in {base_wait_time * (retry_count + 1)} seconds...")
                 time.sleep(base_wait_time * (retry_count + 1))
                 retry_count += 1
                 continue

            # Other client errors (e.g., 404 Not Found) - don't retry immediately
            elif response.status_code >= 400:
                 logger.warning(f"{service_name} client error ({response.status_code}): {url} - {response.text[:200]}")
                 # Consider if specific 4xx errors should trigger retries or specific handling
                 return response # Return the error response without retrying

        except requests.exceptions.Timeout as e:
            logger.warning(f"{service_name} request timed out: {e}. Retrying in {base_wait_time * (retry_count + 1)} seconds...")
            time.sleep(base_wait_time * (retry_count + 1))
            retry_count += 1
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"{service_name} request error: {e}. Retrying in {base_wait_time * (retry_count + 1)} seconds...")
            time.sleep(base_wait_time * (retry_count + 1))
            retry_count += 1
            continue
        except Exception as e: # Catch unexpected errors
            logger.error(f"Unexpected error during {service_name} request: {e}", exc_info=True)
            # Decide if retry makes sense for unexpected errors
            time.sleep(base_wait_time * (retry_count + 1))
            retry_count += 1
            continue

    logger.error(f"Failed {service_name} request to {url} after {max_retries} retries.")
    return None # Indicate failure


# --- Elasticsearch Functions (Adapted from upload_bulk_to_ES.py) ---
def connect_elasticsearch(host, port, user, password):
    """Establish connection to Elasticsearch."""
    logger.info(f"Connecting to Elasticsearch at {host}:{port}")
    try:
        es = Elasticsearch(
            [f'http://{host}:{port}'],
            http_auth=(user, password),
            timeout=60, # Increase default timeout
            max_retries=3,
            retry_on_timeout=True
        )
        if es.ping():
            logger.info("Successfully connected to Elasticsearch")
            return es
        else:
            logger.error("Failed to connect to Elasticsearch (ping failed)")
            return None
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        return None

def fetch_papers_to_enrich(es, index_name, min_citation, max_citation, batch_size=100):
    """
    Fetches papers from Elasticsearch that need enrichment using the scan helper.
    Filters by citation count and missing abstract or pdf url.
    """
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
                "filter": [ # Using filter context for non-scoring queries
                    {
                        "bool": {
                            "should": [
                                { "bool": { "must_not": { "exists": { "field": "abstract" } } } },
                                { "term": { "abstract": "" } }, # Also check for empty string abstract
                                { "bool": { "must_not": { "exists": { "field": "openAccessPdf.url" } } } }
                            ],
                            "minimum_should_match": 1
                        }
                    }
                ]
            }
        },
        "_source": ["paperId", "title", "externalIds", "abstract", "openAccessPdf"] # Fetch only needed fields
    }

    logger.info(f"Fetching papers from index '{index_name}' with citation range [{min_citation}, {max_citation}] missing abstract or PDF URL...")

    # Use scan helper for efficient scrolling
    try:
        # Note: scan returns a generator
        scan_generator = scan(
            client=es,
            index=index_name,
            query=query,
            size=batch_size, # Number of docs per shard to retrieve per scroll
            scroll='5m' # Keep scroll context alive for 5 minutes
        )
        return scan_generator
    except Exception as e:
        logger.error(f"Error fetching papers using scan: {e}")
        return None # Indicate failure

# --- Enrichment Functions (Placeholders - Implement details later) ---

def enrich_from_semantic_scholar(paper_id):
    """Fetch details from Semantic Scholar Single Paper API."""
    if not paper_id:
        return None
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=abstract,isOpenAccess,openAccessPdf"
    # Ideally, add API key header if available
    headers = {} # Add {"x-api-key": YOUR_API_KEY} if you have one
    response = make_request_with_retry('semantic_scholar', 'get', url, headers=headers)
    if response and response.status_code == 200:
        try:
            data = response.json()
            # Extract relevant fields
            update_data = {}
            if data.get('abstract'):
                update_data['abstract'] = data['abstract']

            # Check for PDF URL and verify it ends with .pdf
            s2_pdf_data = data.get('openAccessPdf')
            if data.get('isOpenAccess') and s2_pdf_data and s2_pdf_data.get('url'):
                s2_pdf_url = s2_pdf_data['url']
                if s2_pdf_url and isinstance(s2_pdf_url, str) and s2_pdf_url.lower().endswith('.pdf'):
                    update_data['openAccessPdf'] = {
                        'url': s2_pdf_url,
                        'status': s2_pdf_data.get('status', 'oa_found_s2') # Added source status
                    }
                else:
                    logger.debug(f"URL from S2 ({s2_pdf_url}) is not a .pdf for paper {paper_id}")

            return update_data if update_data else None
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from Semantic Scholar for {paper_id}")
            return None
    elif response and response.status_code == 404:
         logger.info(f"Paper {paper_id} not found in Semantic Scholar Single API (404).")
         return None
    # Log other errors if needed based on make_request_with_retry's return
    return None

def reconstruct_abstract_from_oa(inverted_index):
    """Reconstruct plain text abstract from OpenAlex inverted index. (Copied from pyalex_script.py)"""
    if not inverted_index:
        return None
    try:
        word_positions = [(word, pos) for word, positions in inverted_index.items() for pos in positions]
        # Handle potential non-integer positions if necessary
        sorted_words = sorted(word_positions, key=lambda x: int(x[1]))
        return " ".join(word for word, pos in sorted_words)
    except Exception as e:
        logger.error(f"Error reconstructing abstract: {e} - Index: {inverted_index}")
        return None

def enrich_from_openalex(doi, title=None):
    """Fetch details from OpenAlex API using DOI or title."""
    if not doi and not title:
        return None

    work = None
    try:
        if doi:
            logger.debug(f"Querying OpenAlex with DOI: {doi}")
            # Ensure DOI is the short form if needed, OpenAlex often handles full URLs
            doi_cleaned = doi.replace("https://doi.org/", "")
            _respect_rate_limit('openalex') # Apply rate limit before pyalex call
            results = Works().filter(doi=doi_cleaned).get()
            if results:
                work = results[0]
        elif title:
            # Fallback: Search by title (less reliable)
            # Consider adding author matching for better accuracy if needed
            logger.debug(f"Querying OpenAlex with Title: {title[:50]}...")
            _respect_rate_limit('openalex') # Apply rate limit before pyalex call
            results = Works().filter(title={'search': title}).get()
            # TODO: Add logic to check if the found work is a good match (e.g., compare authors if available)
            if results:
                work = results[0] # Taking the first match

        if work:
            update_data = {}
            # Try reconstructing abstract
            abstract_inverted = work.get("abstract_inverted_index")
            if abstract_inverted:
                 abstract_text = reconstruct_abstract_from_oa(abstract_inverted)
                 if abstract_text:
                     update_data['abstract'] = abstract_text

            # Check for open access PDF URL and verify it ends with .pdf
            best_oa_loc = work.get("best_oa_location")
            if isinstance(best_oa_loc, dict):
                oa_pdf_url = best_oa_loc.get("pdf_url")
                if oa_pdf_url and isinstance(oa_pdf_url, str) and oa_pdf_url.lower().endswith('.pdf'):
                    update_data['openAccessPdf'] = {
                        'url': oa_pdf_url,
                        'status': best_oa_loc.get("license") or best_oa_loc.get('version') or 'oa_found_openalex'
                    }
                elif oa_pdf_url: # Log if a URL was found but wasn't a pdf
                     logger.debug(f"URL from OpenAlex ({oa_pdf_url}) is not a .pdf for DOI {doi or title}")


            return update_data if update_data else None

    except requests.exceptions.RequestException as e:
        # Pyalex might raise requests exceptions
        logger.warning(f"OpenAlex request failed: {e}")
        # Implement retry/backoff if pyalex doesn't handle it internally sufficiently
        time.sleep(5) # Simple backoff
    except Exception as e:
        logger.error(f"Error querying OpenAlex: {e}", exc_info=True)

    return None


def enrich_from_unpaywall(doi):
    """Fetch open access details from Unpaywall API using DOI."""
    if not doi:
        return None
    # Ensure DOI is just the identifier, not the full URL
    doi_cleaned = doi.replace("https://doi.org/", "")
    # Select a random email for this request
    selected_email = random.choice(UNPAYWALL_EMAILS)
    url = f"https://api.unpaywall.org/v2/{doi_cleaned}?email={selected_email}"
    response = make_request_with_retry('unpaywall', 'get', url)

    if response and response.status_code == 200:
        try:
            data = response.json()
            oa_location = data.get("best_oa_location")
            if oa_location and oa_location.get("url_for_pdf"):
                upw_pdf_url = oa_location.get("url_for_pdf")
                # Check if the URL ends with .pdf
                if upw_pdf_url and isinstance(upw_pdf_url, str) and upw_pdf_url.lower().endswith('.pdf'):
                    logger.debug(f"Unpaywall found valid PDF URL for DOI {doi_cleaned}")
                    return {
                        'openAccessPdf': {
                            'url': upw_pdf_url,
                            'status': oa_location.get("version") or oa_location.get("license") or 'oa_found_unpaywall'
                        }
                    }
                else:
                    logger.debug(f"URL from Unpaywall ({upw_pdf_url}) doesn't end with .pdf for DOI {doi_cleaned}")
                    return None # Not a PDF URL
            else:
                # logger.debug(f"No OA PDF found via Unpaywall for DOI {doi_cleaned}")
                return None
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from Unpaywall for {doi}")
            return None
    elif response and response.status_code == 404:
         logger.debug(f"DOI {doi_cleaned} not found in Unpaywall (404).")
         return None
    # Log other errors if needed
    return None

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Enrich Elasticsearch paper data.")
    parser.add_argument("--host", default="35.193.245.34", help="Elasticsearch host.")
    parser.add_argument("--port", type=int, default=9200, help="Elasticsearch port.")
    parser.add_argument("--user", default="elastic", help="Elasticsearch user.")
    parser.add_argument("--password", default="UIm79Shaaaii982", help="Elasticsearch password.")
    parser.add_argument("--index", default="semantic_scholar_papers", help="Elasticsearch index name.")
    parser.add_argument("--min-citation", type=int, required=True, help="Minimum citation count.")
    parser.add_argument("--max-citation", type=int, required=True, help="Maximum citation count.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of papers to process in each enrichment batch.")
    parser.add_argument("--update-batch-size", type=int, default=500, help="Number of updates to send to Elasticsearch in bulk.")
    parser.add_argument("--max-papers", type=int, default=1000000, help="Maximum number of papers to process.")


    args = parser.parse_args()

    # Connect to ES
    es = connect_elasticsearch(args.host, args.port, args.user, args.password)
    if not es:
        sys.exit(1)

    # Fetch papers generator
    papers_generator = fetch_papers_to_enrich(es, args.index, args.min_citation, args.max_citation, args.batch_size)
    if not papers_generator:
        logger.error("Failed to get papers generator from Elasticsearch. Exiting.")
        sys.exit(1)


    # Statistics
    stats = {
        "processed": 0,
        "updates_attempted": 0,
        "updates_successful": 0,
        "updates_failed": 0,
        "abstract_found": {"semantic_scholar": 0, "openalex": 0, "total": 0},
        "pdf_url_found": {"semantic_scholar": 0, "openalex": 0, "unpaywall": 0, "total": 0},
        "semantic_scholar_hits": 0,
        "openalex_hits": 0,
        "unpaywall_hits": 0,
        "errors": {"semantic_scholar": 0, "openalex": 0, "unpaywall": 0, "es_update": 0},
        "start_time": time.time()
    }

    update_actions = []

    # Process papers using the generator
    # Wrap generator in tqdm for progress bar - requires knowing total count, which scan doesn't provide easily.
    # We can estimate or just show progress without total.
    logger.info("Starting paper enrichment process...")
    # Use tqdm without total, just showing iterations/rate
    # papers_iterator = tqdm(papers_generator, desc="Processing papers", unit="paper")
    papers_iterator = papers_generator # Use generator directly

    try:
        for doc in papers_iterator:
            # Check if max_papers limit is reached
            if stats["processed"] >= args.max_papers:
                logger.info(f"Reached max_papers limit ({args.max_papers}). Stopping enrichment.")
                break

            stats["processed"] += 1
            paper_id_es = doc['_id'] # Elasticsearch document ID (_id is usually paperId here)
            source = doc['_source']
            needs_update = False
            update_doc = {} # Document containing fields to update

            # --- Check what's missing ---
            missing_abstract = not source.get('abstract')
            pdf_info = source.get('openAccessPdf', {})
            missing_pdf_url = not isinstance(pdf_info, dict) or not pdf_info.get('url')

            # --- Attempt Enrichment ---
            paper_id_s2 = source.get('paperId') # Semantic Scholar ID
            doi = source.get('externalIds', {}).get('DOI')
            title = source.get('title')

            # 1. Semantic Scholar Single API
            if paper_id_s2 and (missing_abstract or missing_pdf_url):
                s2_data = enrich_from_semantic_scholar(paper_id_s2)
                if s2_data:
                    stats["semantic_scholar_hits"] += 1
                    if missing_abstract and 'abstract' in s2_data:
                        update_doc['abstract'] = s2_data['abstract']
                        stats["abstract_found"]["semantic_scholar"] += 1
                        stats["abstract_found"]["total"] += 1
                        missing_abstract = False # Found it
                        needs_update = True
                        logger.debug(f"DEBUG: Title='{title[:50]}...', Source=S2, Found=Abstract, PaperID={paper_id_es}")
                        if missing_pdf_url and 'openAccessPdf' in s2_data:
                             update_doc['openAccessPdf'] = s2_data['openAccessPdf']
                             stats["pdf_url_found"]["semantic_scholar"] += 1
                             stats["pdf_url_found"]["total"] += 1
                             missing_pdf_url = False # Found it
                             needs_update = True
                             logger.debug(f"DEBUG: Title='{title[:50]}...', Source=S2, Found=PDF({s2_data['openAccessPdf']['url']}), PaperID={paper_id_es}")
                    elif s2_data is None: # Simplified error check based on return
                         pass # Error logging now mainly handled by make_request_with_retry
                elif s2_data is None and make_request_with_retry.last_error_type == 'request_error': # Check if enrich failed due to API error
                     stats["errors"]["semantic_scholar"] += 1


            # 2. OpenAlex API (if still missing something)
            if (missing_abstract or missing_pdf_url) and (doi or title):
                 oa_data = enrich_from_openalex(doi, title if not doi else None) # Prefer DOI
                 if oa_data:
                    stats["openalex_hits"] += 1
                    if missing_abstract and 'abstract' in oa_data:
                        update_doc['abstract'] = oa_data['abstract']
                        stats["abstract_found"]["openalex"] += 1
                        stats["abstract_found"]["total"] += 1
                        missing_abstract = False
                        needs_update = True
                        logger.debug(f"DEBUG: Title='{title[:50]}...', Source=OpenAlex, Found=Abstract, PaperID={paper_id_es}")
                    if missing_pdf_url and 'openAccessPdf' in oa_data:
                         # Only update if S2 didn't already find one
                         if 'openAccessPdf' not in update_doc:
                             update_doc['openAccessPdf'] = oa_data['openAccessPdf']
                             stats["pdf_url_found"]["openalex"] += 1
                             stats["pdf_url_found"]["total"] += 1
                             missing_pdf_url = False
                             needs_update = True
                             logger.debug(f"DEBUG: Title='{title[:50]}...', Source=OpenAlex, Found=PDF({oa_data['openAccessPdf']['url']}), PaperID={paper_id_es}")
                 elif oa_data is None and make_request_with_retry.last_error_type == 'request_error': # Approx check
                      stats["errors"]["openalex"] += 1


            # 3. Unpaywall API (if still missing PDF URL)
            if missing_pdf_url and doi:
                 upw_data = enrich_from_unpaywall(doi)
                 if upw_data:
                    stats["unpaywall_hits"] += 1
                    if 'openAccessPdf' in upw_data:
                        # Only update if not already found
                        if 'openAccessPdf' not in update_doc:
                             update_doc['openAccessPdf'] = upw_data['openAccessPdf']
                             stats["pdf_url_found"]["unpaywall"] += 1
                             stats["pdf_url_found"]["total"] += 1
                             missing_pdf_url = False
                             needs_update = True
                             logger.debug(f"DEBUG: Title='{title[:50]}...', Source=Unpaywall, Found=PDF({upw_data['openAccessPdf']['url']}), PaperID={paper_id_es}")
                    elif upw_data is None: # Simplified error check
                        pass # Error logging handled by make_request_with_retry

            # --- Prepare Bulk Update Action ---
            if needs_update and update_doc:
                stats["updates_attempted"] += 1
                action = {
                    "_op_type": "update",
                    "_index": args.index,
                    "_id": paper_id_es,
                    "doc": update_doc
                }
                update_actions.append(action)

            # --- Perform Bulk Update Periodically ---
            if len(update_actions) >= args.update_batch_size:
                logger.info(f"Sending bulk update for {len(update_actions)} papers...")
                try:
                    success_count, errors = bulk(es, update_actions, raise_on_error=False)
                    stats["updates_successful"] += success_count
                    if errors:
                        failed_count = len(errors)
                        stats["updates_failed"] += failed_count
                        stats["errors"]["es_update"] += failed_count
                        logger.warning(f"{failed_count} bulk update errors occurred. First few: {errors[:5]}")
                    logger.info(f"Bulk update results: {success_count} successful, {len(errors)} failed.")
                except Exception as e:
                    failed_count = len(update_actions) # Assume all failed if bulk call raises exception
                    stats["updates_failed"] += failed_count
                    stats["errors"]["es_update"] += failed_count
                    logger.error(f"Error during bulk update: {e}")
                update_actions = [] # Clear actions buffer

            # --- Logging Progress Periodically ---
            if stats["processed"] % 1000 == 0:
                 elapsed_time = time.time() - stats["start_time"]
                 rate = stats["processed"] / elapsed_time if elapsed_time > 0 else 0
                 logger.info(f"Progress: Processed={stats['processed']}, Updated={stats['updates_successful']}, Rate={rate:.1f}/s, "
                             f"Abstracts Found={stats['abstract_found']['total']}, PDFs Found={stats['pdf_url_found']['total']}, "
                             f"Errors={sum(stats['errors'].values())}")

    except Exception as e:
         logger.error(f"Error during enrichment loop: {e}", exc_info=True)
         # Decide whether to attempt final bulk update or exit

    finally:
        # --- Final Bulk Update ---
        if update_actions:
             logger.info(f"Sending final bulk update for {len(update_actions)} papers...")
             try:
                 success_count, errors = bulk(es, update_actions, raise_on_error=False)
                 stats["updates_successful"] += success_count
                 if errors:
                    failed_count = len(errors)
                    stats["updates_failed"] += failed_count
                    stats["errors"]["es_update"] += failed_count
                    logger.warning(f"{failed_count} final bulk update errors occurred. First few: {errors[:5]}")
                 logger.info(f"Final bulk update results: {success_count} successful, {len(errors)} failed.")
             except Exception as e:
                 failed_count = len(update_actions)
                 stats["updates_failed"] += failed_count
                 stats["errors"]["es_update"] += failed_count
                 logger.error(f"Error during final bulk update: {e}")

        # --- Print Final Statistics ---
        end_time = time.time()
        total_time = end_time - stats["start_time"]
        hours, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)

        logger.info("="*50)
        logger.info("ENRICHMENT PROCESS COMPLETE")
        logger.info("="*50)
        logger.info(f"Total time: {int(hours):02d}h {int(mins):02d}m {int(secs):02d}s")
        logger.info(f"Papers processed: {stats['processed']}")
        logger.info(f"Updates attempted: {stats['updates_attempted']}")
        logger.info(f"Updates successful: {stats['updates_successful']}")
        logger.info(f"Updates failed: {stats['updates_failed']}")
        logger.info("-" * 20)
        logger.info("Data Found By Source:")
        logger.info(f"  Semantic Scholar Hits: {stats['semantic_scholar_hits']}")
        logger.info(f"  OpenAlex Hits: {stats['openalex_hits']}")
        logger.info(f"  Unpaywall Hits: {stats['unpaywall_hits']}")
        logger.info("-" * 20)
        logger.info("Abstracts Found:")
        logger.info(f"  via Semantic Scholar: {stats['abstract_found']['semantic_scholar']}")
        logger.info(f"  via OpenAlex: {stats['abstract_found']['openalex']}")
        logger.info(f"  Total: {stats['abstract_found']['total']}")
        logger.info("-" * 20)
        logger.info("PDF URLs Found:")
        logger.info(f"  via Semantic Scholar: {stats['pdf_url_found']['semantic_scholar']}")
        logger.info(f"  via OpenAlex: {stats['pdf_url_found']['openalex']}")
        logger.info(f"  via Unpaywall: {stats['pdf_url_found']['unpaywall']}")
        logger.info(f"  Total: {stats['pdf_url_found']['total']}")
        logger.info("-" * 20)
        logger.info("Errors Encountered:")
        logger.info(f"  Semantic Scholar API: {stats['errors']['semantic_scholar']}")
        logger.info(f"  OpenAlex API: {stats['errors']['openalex']}")
        logger.info(f"  Unpaywall API: {stats['errors']['unpaywall']}")
        logger.info(f"  Elasticsearch Update: {stats['errors']['es_update']}")
        logger.info("="*50)

if __name__ == "__main__":
    # Add a check for the last error type to the request function
    # This is a bit hacky; a class-based approach might be cleaner
    make_request_with_retry.last_error_type = None
    main() 
