import requests
import json
import time
import os
import argparse
import logging
from tqdm import tqdm

class SemanticScholarBulkSearch:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key=None, output_dir="./search_results", log_file="search_logs.log"):
        """Initialize the search client with optional API key and output directory."""
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Rate limiting parameters
        self.requests_per_second = 5 if api_key else 1
        self.last_request_time = 0
        
        # Set up logging
        self.setup_logging(log_file)
        self.logger = logging.getLogger('semantic_scholar')
    
    def setup_logging(self, log_file):
        """Set up logging to file and console."""
        log_path = os.path.join(self.output_dir, log_file)
        
        # Create logger
        logger = logging.getLogger('semantic_scholar')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicate logs
        if logger.handlers:
            logger.handlers = []
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path)  # Use log_path with correct directory
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed the rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # If we need to wait to respect the rate limit
        if time_since_last_request < 1.0 / self.requests_per_second:
            sleep_time = (1.0 / self.requests_per_second) - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def check_internet_connection(self, timeout=5):
        """
        Check if there is an active internet connection.
        Returns True if connected, False otherwise.
        """
        try:
            # Try connecting to a reliable server with a short timeout
            requests.get("https://www.google.com", timeout=timeout)
            return True
        except requests.exceptions.RequestException:
            return False
    
    def make_request_with_retry(self, method, url, **kwargs):
        """
        Make an HTTP request with retry logic for handling errors.
        
        Args:
            method: HTTP method (e.g., 'get', 'post')
            url: URL to request
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            requests.Response: The response object if successful
        """
        retry_count = 0
        max_wait_time = 300  # Maximum wait time between retries (5 minutes)
        
        while True:
            # If we have network issues, check for internet connectivity
            if retry_count > 3:
                if not self.check_internet_connection():
                    self.logger.warning("No internet connection detected. Waiting for connectivity...")
                    # Wait for a longer time when no internet connection
                    wait_time = min(30 * (retry_count - 2), max_wait_time)
                    self.logger.info(f"Will retry in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
            
            try:
                self._respect_rate_limit()
                
                self.logger.info(f"Making {method.upper()} request to {url}")
                
                if method.lower() == 'get':
                    response = requests.get(url, **kwargs)
                elif method.lower() == 'post':
                    response = requests.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for success
                if response.status_code == 200:
                    self.logger.info(f"Request successful (status {response.status_code})")
                    return response
                
                # Handle errors
                if response.status_code >= 400:
                    self.logger.warning(f"Request failed with status code {response.status_code}: {response.text}")
                    
                    # Specific handling for rate limiting
                    if response.status_code == 429:
                        self.logger.warning("Rate limit exceeded. Waiting longer...")
                        wait_time = 30  # Wait longer for rate limit errors
                    else:
                        wait_time = min(10 * (retry_count + 1), max_wait_time)
                    
                    self.logger.info(f"Retrying in {wait_time} seconds (attempt {retry_count+1})...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
            
            except requests.exceptions.RequestException as e:
                # Network-related errors - retry indefinitely with backoff
                is_network_error = any(err_type in str(e) for err_type in [
                    "NameResolutionError", 
                    "ConnectionError", 
                    "Temporary failure",
                    "timeout",
                    "ConnectTimeoutError",
                    "ReadTimeoutError"
                ])
                
                # Save current continuation token and page offset if available in kwargs
                # This is done inside an exception so it's only saved on errors
                if 'params' in kwargs and 'token' in kwargs['params']:
                    token = kwargs['params']['token']
                    page_offset = kwargs['params'].get('offset', 0)
                    self.save_continuation_token(token, page_offset)
                    self.logger.info(f"Saved continuation token during error for recovery")
                
                # Use exponential backoff for network errors, cap at max_wait_time
                wait_time = min(2 ** min(retry_count, 8), max_wait_time)
                self.logger.error(f"Request error: {str(e)}")
                self.logger.info(f"Retrying after error in {wait_time} seconds (attempt {retry_count+1})...")
                time.sleep(wait_time)
                retry_count += 1
                continue
    
    def save_continuation_token(self, token, page_offset):
        """Save continuation token and page offset to file for resumption capability."""
        token_file = os.path.join(self.output_dir, "continuation_token.txt")
        backup_file = os.path.join(self.output_dir, f"continuation_token.backup.{int(time.time())}.txt")
        
        content = f"{token}\n{page_offset}"
        
        # Try to save to primary file
        try:
            with open(token_file, 'w') as f:
                f.write(content)
            self.logger.info(f"Saved continuation token: {token}, offset: {page_offset}")
            return True
        except IOError as e:
            self.logger.error(f"Failed to save token to main file: {str(e)}")
            # If primary save fails, try backup
            try:
                with open(backup_file, 'w') as f:
                    f.write(content)
                self.logger.info(f"Saved continuation token to backup file")
                return True
            except IOError as e:
                self.logger.error(f"Failed to save token to backup file: {str(e)}")
                return False
    
    def bulk_search_papers(self, query=None, fields=None, sort=None, publication_types=None, 
                      open_access_pdf=False, min_citation_count=None, 
                      publication_date_or_year=None, year=None, venue=None, 
                      fields_of_study=None, max_papers=None, resume_token=None,
                      page_offset=0,
                      output_file="search_results.json"):
        """
        Search for papers using the bulk search API with resumption support.
        
        Args:
            query (str, optional): Text query to search for. If None, use "*" to match everything.
            fields (list): List of fields to include in the response
            sort (str): Sort order (e.g., "citationCount:desc")
            publication_types (list): List of publication types to filter by
            open_access_pdf (bool): Whether to only include papers with open access PDFs
            min_citation_count (int): Minimum citation count
            publication_date_or_year (str): Publication date range (e.g., "2019-03:2020-06")
            year (str): Publication year range (e.g., "2015-2020")
            venue (list): List of venues to filter by
            fields_of_study (list): List of fields of study to filter by
            max_papers (int): Maximum number of papers to retrieve (None for all)
            resume_token (str): Token to resume search from
            page_offset (int): Page offset for pagination
            output_file (str): Filename to save results incrementally
            
        Returns:
            list: List of paper data dictionaries
        """
        url = f"{self.BASE_URL}/paper/search/bulk"
        
        # Build query parameters
        # If no query is provided, use "*" as a wildcard to match everything
        params = {"query": query if query else "*"}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        if sort:
            params["sort"] = sort
        
        if publication_types:
            params["publicationTypes"] = ",".join(publication_types)
        
        if open_access_pdf:
            params["openAccessPdf"] = "true"
        
        if min_citation_count:
            params["minCitationCount"] = str(min_citation_count)
        
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        
        if year:
            params["year"] = year
        
        if venue:
            params["venue"] = ",".join(venue)
        
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        
        # Initialize results
        all_papers = []
        continuation_token = resume_token
        current_page_offset = page_offset
        pbar = None
        output_path = os.path.join(self.output_dir, output_file)
        
        # Check if we're resuming and load existing data
        if resume_token and os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    all_papers = json.load(f)
                self.logger.info(f"Resuming search from token with current page offset {current_page_offset}. Loaded {len(all_papers)} existing papers.")
            except Exception as e:
                self.logger.warning(f"Error loading existing results: {str(e)}. Starting fresh.")
                all_papers = []
        
        # Keep fetching until we have all papers or reach the max_papers limit
        batch_count = 0
        missing_token_retries = 0
        max_missing_token_retries = 125  # Number of times to retry when token is missing
        total_papers_expected = None
        
        while True:
            batch_count += 1
            # Add continuation token if we have one, otherwise use page offset
            if continuation_token and continuation_token != "NO_TOKEN":
                params["token"] = continuation_token
                if "offset" in params:
                    del params["offset"]  # Remove offset parameter when using token
                self.logger.info(f"Using continuation token: {continuation_token}")
            else:
                if "token" in params:
                    del params["token"]  # Remove token parameter when using offset
                params["offset"] = current_page_offset
                self.logger.info(f"Using page offset: {current_page_offset}")
            
            self.logger.info(f"Fetching batch {batch_count} of papers...")
            
            try:
                # Make the request with retry logic
                response = self.make_request_with_retry('get', url, params=params, headers=self.headers)
                
                # Parse response
                result = response.json()
                
                # Initialize progress bar if this is the first request or we don't have one yet
                if pbar is None:
                    total_papers_expected = result.get("total", 0)
                    total = min(total_papers_expected, max_papers) if max_papers else total_papers_expected
                    pbar = tqdm(total=total, desc="Fetching papers")
                    pbar.update(len(all_papers))  # Update with papers we've already processed
                    self.logger.info(f"Found {total_papers_expected} papers. Will fetch {'all' if not max_papers else max_papers}.")
                
                # Add papers to our results
                papers = result.get("data", [])
                if len(papers) > 0:
                    all_papers.extend(papers)
                    papers_fetched = len(all_papers)
                    pbar.update(len(papers))
                    
                    self.logger.info(f"Fetched {len(papers)} papers in this batch. Total: {papers_fetched} papers")
                    
                    # Increment page offset for next request
                    current_page_offset += len(papers)
                    
                    # Incrementally save results after each batch
                    self.save_results(all_papers, output_file, append=False)
                    self.logger.info(f"Saved {papers_fetched} papers to {output_path}")
                    
                    # Reset missing token retries because we got valid data
                    missing_token_retries = 0
                else:
                    self.logger.info("No papers in this batch. Might be a temporary issue.")
                    missing_token_retries += 1
                
                # Update continuation token for next request
                continuation_token = result.get("token")
                
                # Always save the current state for potential recovery
                self.save_continuation_token(continuation_token if continuation_token else "NO_TOKEN", current_page_offset)
                
                # Log the continuation token for debugging
                if continuation_token:
                    self.logger.info(f"Next continuation token: {continuation_token}, page offset: {current_page_offset}")
                    # Reset missing token retries because we got a token
                    missing_token_retries = 0
                else:
                    self.logger.warning("No continuation token received.")
                    missing_token_retries += 1
                    
                    # Check if we've got all expected papers
                    if total_papers_expected and len(all_papers) >= total_papers_expected:
                        self.logger.info(f"Fetched all {total_papers_expected} papers. Search is complete.")
                        break
                    
                    # If we're missing tokens but haven't reached the expected total
                    if missing_token_retries <= max_missing_token_retries:
                        self.logger.info(f"Will retry using page offset (retry {missing_token_retries}/{max_missing_token_retries})")
                        time.sleep(10)  # Wait before retrying
                        continue
                    else:
                        self.logger.warning(f"Missing continuation token after {max_missing_token_retries} retries.")
                
                # Stop if we've reached our limit or exceeded max missing token retries
                if (max_papers and papers_fetched >= max_papers) or (missing_token_retries > max_missing_token_retries and not continuation_token):
                    break
                
            except Exception as e:
                # Handle any unexpected errors gracefully
                self.logger.error(f"Unexpected error during search: {str(e)}")
                
                # Save current state for recovery
                self.save_continuation_token(continuation_token if continuation_token else "NO_TOKEN", current_page_offset)
                self.logger.info(f"Saved state for recovery: token={continuation_token}, offset={current_page_offset}")
                
                # Wait and retry
                wait_time = 30
                self.logger.info(f"Will retry in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
        
        if pbar:
            pbar.close()
        
        # Check if we got all the expected papers
        if total_papers_expected and len(all_papers) < total_papers_expected:
            completion_percentage = (len(all_papers) / total_papers_expected) * 100
            self.logger.warning(f"Search ended at {completion_percentage:.2f}% completion. Got {len(all_papers)} out of {total_papers_expected} papers.")
            
            # Save the last state for possible manual recovery
            if continuation_token:
                self.save_continuation_token(continuation_token, current_page_offset)
            else:
                # If we don't have a token, save the current page offset
                self.save_continuation_token("NO_TOKEN", current_page_offset)
                
            self.logger.info(f"Saved final state for manual recovery if needed.")
        
        # Trim results if we fetched more than max_papers
        if max_papers and len(all_papers) > max_papers:
            all_papers = all_papers[:max_papers]
            # Save the final trimmed results
            self.save_results(all_papers, output_file, append=False)
        
        self.logger.info(f"Search completed. Retrieved {len(all_papers)} papers.")
        return all_papers
    
    def save_results(self, data, filename, append=False):
        """
        Save results to a JSON file.
        
        Args:
            data: Data to save
            filename: Output filename
            append: Whether to append to existing file or overwrite
        """
        filepath = os.path.join(self.output_dir, filename)
        
        if append and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                if isinstance(existing_data, list) and isinstance(data, list):
                    data = existing_data + data
                elif isinstance(existing_data, dict) and isinstance(data, dict):
                    existing_data.update(data)
                    data = existing_data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(f"Error reading existing file for append: {str(e)}")
                # Continue with writing new data
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return filepath
        except IOError as e:
            self.logger.error(f"Error saving data to {filepath}: {str(e)}")
            # Try saving to a backup file
            backup_path = f"{filepath}.backup.{int(time.time())}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved data to backup file: {backup_path}")
            return backup_path

def load_continuation_token(output_dir):
    """
    Load continuation token and page offset from file if it exists.
    """
    token_file = os.path.join(output_dir, "continuation_token.txt")
    
    # First try the main token file
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    token = lines[0]  # First line is the token
                    page_offset = int(lines[1])  # Second line is the page offset
                    
                    # Handle special case for NO_TOKEN
                    if token == "NO_TOKEN":
                        return None, page_offset
                    
                    return token, page_offset
        except (IOError, ValueError) as e:
            print(f"Error reading main token file: {str(e)}")
    
    # If main file doesn't exist or had an error, try backup files
    backup_files = [f for f in os.listdir(output_dir) 
                   if f.startswith("continuation_token.backup") and f.endswith(".txt")]
    
    if backup_files:
        # Sort by timestamp (newest first)
        backup_files.sort(reverse=True)
        
        for backup in backup_files:
            try:
                with open(os.path.join(output_dir, backup), 'r') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        token = lines[0]  # First line is the token
                        page_offset = int(lines[1])  # Second line is the page offset
                        
                        # Handle special case for NO_TOKEN
                        if token == "NO_TOKEN":
                            return None, page_offset
                        
                        return token, page_offset
            except (IOError, ValueError) as e:
                print(f"Error reading backup file {backup}: {str(e)}")
                continue
    
    # If no valid token found
    return None, 0

def main():
    parser = argparse.ArgumentParser(description="Search for papers using Semantic Scholar bulk search API")
    parser.add_argument("--query", help="Text query to search for (optional)")
    parser.add_argument("--fields", help="Comma-separated list of fields to include in the response")
    parser.add_argument("--sort", help="Sort order (e.g., 'citationCount:desc')")
    parser.add_argument("--publication-types", help="Comma-separated list of publication types")
    parser.add_argument("--open-access-pdf", action="store_true", help="Only include papers with open access PDFs")
    parser.add_argument("--min-citation-count", type=int, help="Minimum citation count")
    parser.add_argument("--publication-date-or-year", help="Publication date range (e.g., '2019-03:2020-06')")
    parser.add_argument("--year", help="Publication year range (e.g., '2015-2020')")
    parser.add_argument("--venue", help="Comma-separated list of venues")
    parser.add_argument("--fields-of-study", help="Comma-separated list of fields of study")
    parser.add_argument("--max-papers", type=int, help="Maximum number of papers to retrieve")
    parser.add_argument("--output", default="search_results.json", help="Output filename")
    parser.add_argument("--api-key", help="API key for Semantic Scholar")
    parser.add_argument("--output-dir", default="./search_results", help="Directory to save results")
    parser.add_argument("--log-file", default="search_logs.log", help="Log file name")
    parser.add_argument("--auto-recover", action="store_true", help="Automatically recover from last saved token if available")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved token (explicit flag)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up resume information
    resume_token = None
    page_offset = 0
    
    # Check for existing tokens if auto-recover or resume is enabled
    if args.auto_recover or args.resume:
        resume_token, page_offset = load_continuation_token(args.output_dir)
        if resume_token or page_offset > 0:
            print(f"Resuming search with {'token: ' + resume_token if resume_token else 'page offset: ' + str(page_offset)}")
        else:
            print("No continuation data found. Starting a new search.")
    
    # Split comma-separated values into lists
    fields = args.fields.split(",") if args.fields else ["title", "abstract", "authors", "venue", "year", "citationCount", "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes", "externalIds", "openAccessPdf", "publicationDate", "url"]
    publication_types = args.publication_types.split(",") if args.publication_types else None
    venue = args.venue.split(",") if args.venue else None
    fields_of_study = args.fields_of_study.split(",") if args.fields_of_study else None
    
    # Initialize the search client
    searcher = SemanticScholarBulkSearch(api_key=args.api_key, output_dir=args.output_dir, log_file=args.log_file)
    searcher.logger.info("Starting Semantic Scholar Bulk Search")
    
    if resume_token or page_offset > 0:
        searcher.logger.info(f"Resuming search with {'token: ' + resume_token if resume_token else 'page offset: ' + str(page_offset)}")
    else:
        searcher.logger.info("Starting a new search.")
    
    # Search for papers
    papers = searcher.bulk_search_papers(
        query=args.query,
        fields=fields,
        sort=args.sort,
        publication_types=publication_types,
        open_access_pdf=args.open_access_pdf,
        min_citation_count=args.min_citation_count,
        publication_date_or_year=args.publication_date_or_year,
        year=args.year,
        venue=venue,
        fields_of_study=fields_of_study,
        max_papers=args.max_papers,
        resume_token=resume_token,
        page_offset=page_offset,
        output_file=args.output
    )
    
    searcher.logger.info("Search process completed successfully")

if __name__ == "__main__":
    main()
