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
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
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
        max_retries = 10
        retry_count = 0
        infinite_network_retries = True  # Set to True to keep retrying network errors indefinitely
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
                        retry_count += 1
                        wait_time = 30  # Wait longer for rate limit errors
                    else:
                        retry_count += 1
                        wait_time = 10
                    
                    self.logger.info(f"Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
            
            except requests.exceptions.RequestException as e:
                # Network-related errors that should retry indefinitely
                is_network_error = any(err_type in str(e) for err_type in [
                    "NameResolutionError", 
                    "ConnectionError", 
                    "Temporary failure",
                    "timeout",
                    "ConnectTimeoutError",
                    "ReadTimeoutError"
                ])
                
                if is_network_error and infinite_network_retries:
                    # Use exponential backoff for network errors
                    wait_time = min(2 ** retry_count, max_wait_time)  # Cap at max_wait_time
                    self.logger.error(f"Network error: {str(e)}")
                    self.logger.info(f"Retrying after network error in {wait_time} seconds (attempt {retry_count+1})...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                else:
                    # Non-network errors use standard retry logic
                    retry_count += 1
                    if retry_count < max_retries:
                        self.logger.error(f"Request error: {str(e)}")
                        self.logger.info(f"Retrying in 10 seconds (attempt {retry_count}/{max_retries})...")
                        time.sleep(10)
                        continue
            
            # If we've exhausted retries for non-network errors
            if retry_count >= max_retries:
                self.logger.error(f"Failed after {retry_count} retries. Waiting 30 minutes before continuing...")
                
                # Sleep for 30 minutes
                minutes_30 = 30 * 60
                self.logger.info(f"Sleeping for 30 minutes!!")
                time.sleep(minutes_30)
                retry_count = 0  # Reset retry count after sleep
            
            # Try one more time after 30 minutes
            self._respect_rate_limit()
            if method.lower() == 'get':
                return requests.get(url, **kwargs)
            else:
                return requests.post(url, **kwargs)
    
    def save_continuation_token(self, token, page_offset):
        """Save continuation token and page offset to file for resumption capability."""
        token_file = os.path.join(self.output_dir, "continuation_token.txt")
        backup_file = os.path.join(self.output_dir, f"continuation_token.backup.{int(time.time())}.txt")
        
        # Try to save to primary file
        try:
            with open(token_file, 'w') as f:
                f.write(f"{token}\n{page_offset}")
            return True
        except IOError as e:
            # If primary save fails, try backup
            try:
                with open(backup_file, 'w') as f:
                    f.write(f"{token}\n{page_offset}")
                return True
            except IOError:
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
        params = {"query": query if query else "*", "offset": page_offset}
        
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
                    self.logger.info(f"Resuming search with {len(all_papers)} papers already collected at page offset {current_page_offset}")
            except json.JSONDecodeError:
                self.logger.warning(f"Could not load existing data from {output_path}. Starting fresh.")
                all_papers = []
        
        # Keep fetching until we have all papers or reach the max_papers limit
        batch_count = 0
        while True:
            batch_count += 1
            # Add continuation token if we have one
            if continuation_token:
                params["token"] = continuation_token
                self.logger.info(f"Using continuation token: {continuation_token} with page offset: {current_page_offset}")
            else:
                params["offset"] = current_page_offset
                self.logger.info(f"Using page offset: {current_page_offset}")
            
            self.logger.info(f"Fetching batch {batch_count} of papers...")
            
            # Make the request with retry logic
            response = self.make_request_with_retry('get', url, params=params, headers=self.headers)
            
            # Check for errors (this should rarely happen due to retry logic)
            if response.status_code != 200:
                self.logger.error(f"Error: {response.status_code} - {response.text}")
                break
            
            # Parse response
            result = response.json()
            
            # Initialize progress bar if this is the first request
            if pbar is None:
                total = min(result["total"], max_papers) if max_papers else result["total"]
                pbar = tqdm(total=total, desc="Fetching papers")
                self.logger.info(f"Found {result['total']} papers. Will fetch {'all' if not max_papers else max_papers}.")
            
            # Add papers to our results
            papers = result["data"]
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
            else:
                self.logger.info("No papers in this batch. Might have reached the end of results.")
            
            # Check if we need to continue
            continuation_token = result.get("token")
            
            # Log the continuation token and page offset for resumption capability
            if continuation_token:
                self.logger.info(f"Next continuation token: {continuation_token}, page offset: {current_page_offset}")
                
                # Save token and page offset to a file for resumption
                if self.save_continuation_token(continuation_token, current_page_offset):
                    self.logger.info(f"Saved continuation token and page offset to continuation_token.txt")
                else:
                    self.logger.warning("Failed to save continuation token")
            else:
                self.logger.info("No more continuation tokens. Search is complete.")
            
            # Stop if we've reached our limit or there are no more papers
            if not continuation_token or (max_papers and papers_fetched >= max_papers):
                break
        
        if pbar:
            pbar.close()
        
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
    Also checks for backup token files if the main one fails.
    """
    token_file = os.path.join(output_dir, "continuation_token.txt")
    
    # First try the main token file
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                lines = f.read().strip().split('\n')
                token = lines[0] if lines else None
                page_offset = int(lines[1]) if len(lines) > 1 else 0
                return token, page_offset
        except (IOError, ValueError):
            # If main file fails, look for backups
            pass
    
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
                    token = lines[0] if lines else None
                    page_offset = int(lines[1]) if len(lines) > 1 else 0
                    return token, page_offset
            except (IOError, ValueError):
                continue
    
    # If no valid token found
    return None, 0

def auto_recovery(args):
    """
    Automatically recover from a failed or interrupted search by using the last saved token.
    Returns updated arguments with resume information if a token is found.
    """
    if not os.path.exists(args.output_dir):
        return args
    
    # Only auto-recover if not explicitly resuming already
    if not args.resume:
        token, page_offset = load_continuation_token(args.output_dir)
        if token:
            print(f"Found a continuation token from a previous run. Auto-recovering...")
            print(f"Token: {token}")
            print(f"Page offset: {page_offset}")
            
            # Ask for confirmation if in interactive mode
            if not args.auto_recover:
                response = input("Do you want to resume from this point? (y/n): ").lower()
                if response != 'y':
                    print("Not resuming. Starting a new search.")
                    return args
            
            # Update args to enable resumption
            args.resume = True
            return args
    
    return args

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
    parser.add_argument("--resume", action="store_true", help="Resume from last known point if available")
    parser.add_argument("--auto-recover", action="store_true", help="Automatically recover from last saved token without asking")
    args = parser.parse_args()
    
    # Apply auto-recovery if enabled
    args = auto_recovery(args)
    
    # Split comma-separated values into lists
    fields = args.fields.split(",") if args.fields else ["title", "abstract", "authors", "venue", "year", "citationCount", "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes", "externalIds", "openAccessPdf", "publicationDate", "url"]
    publication_types = args.publication_types.split(",") if args.publication_types else None
    venue = args.venue.split(",") if args.venue else None
    fields_of_study = args.fields_of_study.split(",") if args.fields_of_study else None
    
    # Initialize the search client
    searcher = SemanticScholarBulkSearch(api_key=args.api_key, output_dir=args.output_dir, log_file=args.log_file)
    searcher.logger.info("Starting Semantic Scholar Bulk Search")
    
    # Check if we should resume
    resume_token = None
    page_offset = 0
    if args.resume:
        resume_token, page_offset = load_continuation_token(args.output_dir)
        if resume_token:
            searcher.logger.info(f"Resuming search with token: {resume_token}, page offset: {page_offset}")
        else:
            searcher.logger.info("No continuation token found. Starting a new search.")
    
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
