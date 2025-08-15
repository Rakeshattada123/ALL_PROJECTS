# main.py

import asyncio
import csv
import json
import os
import re
from datetime import datetime
from typing import List, Dict

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, BackgroundTasks

# --- FastAPI App Initialization ---
app = FastAPI(
    title="BBC Sports Web Scraper API",
    description="An API to scrape sports articles from BBC, save links, and process article metadata.",
    version="1.0.0"
)

# --- Configuration (from the original notebook) ---
BASE_URL = "https://www.bbc.com"
CATEGORY_URL = f"{BASE_URL}/sport"
WEBSITE_NAME = "BBC"
CATEGORY_NAME = "Sports"
CSV_FILENAME = "bbc_sports_articles.csv"
JSON_OUTPUT_DIR = "article_metadata"

# --- Helper Functions (from the original notebook) ---
def clean_filename(title: str) -> str:
    """Cleans a string to be a valid filename."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.replace(" ", "_")
    return title[:100]

# Ensure the output directory for JSON files exists on startup
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# --- API Endpoints ---

@app.post("/scrape-links", summary="Scrape and Save Article Links", tags=["Scraping"])
async def scrape_links_endpoint():
    """
    Scrapes the BBC Sports category page for unique article URLs
    and saves them to a CSV file on the server.
    
    This corresponds to **Task 1** from the notebook.
    """
    print("--- Task 1: Scraping article links from BBC Sports ---")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # The client will now automatically follow redirects (e.g., 301, 302)
    async with httpx.AsyncClient(headers=headers, timeout=20.0, follow_redirects=True) as client: # <--- CHANGE HERE
        try:
            response = await client.get(CATEGORY_URL)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error fetching category page {CATEGORY_URL}: {e}")

    soup = BeautifulSoup(response.content, 'html.parser')
    article_urls = set()

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # The regex from your notebook finds URLs ending in numbers, which are being redirected.
        # It's better to look for the new format as well. Let's keep the original logic for now,
        # as `follow_redirects` will solve the processing issue.
        if href.startswith('/sport/') and re.search(r'(\d+$|/c[a-zA-Z0-9]{12,}[a-zA-Z0-9])$', href):
            # Construct a full URL, handling both absolute and relative paths
            if href.startswith('http'):
                full_url = href
            else:
                full_url = f"{BASE_URL}{href}"
            article_urls.add(full_url)

    if not article_urls:
        return {"message": "No article links found. The website structure might have changed.", "links_found": 0}

    sorted_urls = sorted(list(article_urls))
    
    try:
        with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['website', 'category', 'article_url'])
            for url in sorted_urls:
                writer.writerow([WEBSITE_NAME, CATEGORY_NAME, url])
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write to CSV file: {e}")

    return {
        "message": f"Successfully saved {len(sorted_urls)} article links.",
        "file_saved_as": CSV_FILENAME,
        "links_found": len(sorted_urls)
    }


async def process_single_article(article_url: str, client: httpx.AsyncClient) -> dict:
    """Helper function to scrape metadata from a single article URL."""
    try:
        # The client passed in here is already configured to follow redirects
        response = await client.get(article_url)
        response.raise_for_status()
        await asyncio.sleep(1) # Be respectful to the server
    except httpx.RequestError as e:
        print(f"Could not fetch article {article_url}. Error: {e}")
        return {"url": article_url, "status": "error", "detail": str(e)}

    # The `response.url` attribute will hold the FINAL URL after any redirects
    final_url = str(response.url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extraction logic from the notebook, made more robust
    title_tag = soup.find('h1') # A more generic h1 lookup might be more robust
    if not title_tag:
        title_tag = soup.find('h1', id='main-heading')
    title = title_tag.get_text(strip=True) if title_tag else "N/A"

    summary_tag = soup.find('meta', property='og:description')
    summary = summary_tag['content'] if summary_tag and 'content' in summary_tag.attrs else "N/A"

    publish_date_tag = soup.find('time', {'data-testid': 'timestamp'})
    publish_date = publish_date_tag['datetime'] if publish_date_tag and 'datetime' in publish_date_tag.attrs else "N/A"

    article_image_tag = soup.find('meta', property='og:image')
    article_image = article_image_tag['content'] if article_image_tag and 'content' in article_image_tag.attrs else "N/A"

    article_body = soup.find('article')
    if article_body:
        content_paragraphs = article_body.find_all('p') # More general 'p' tag search
        article_content = "\n".join([p.get_text(strip=True) for p in content_paragraphs])
    else:
        article_content = "N/A"
    
    # If the article content is still empty, try the other component selector
    if not article_content or article_content == "N/A":
        if article_body:
            content_paragraphs = article_body.find_all('div', {'data-component': 'text-block'})
            article_content = "\n".join([p.get_text(strip=True) for p in content_paragraphs])
        else:
            article_content = "N/A"

    metadata = {
        'url': final_url, # Store the final URL
        'title': title,
        'summary': summary,
        'publish_date': publish_date,
        'article_image': article_image,
        'article_content': article_content
    }

    if metadata['title'] != "N/A":
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        clean_title = clean_filename(metadata['title'])
        json_filename = f"{WEBSITE_NAME}_{CATEGORY_NAME}_{clean_title}_{timestamp}.json"
        json_filepath = os.path.join(JSON_OUTPUT_DIR, json_filename)

        try:
            with open(json_filepath, 'w', encoding='utf-8') as jsonfile:
                json.dump(metadata, jsonfile, indent=4, ensure_ascii=False)
            return {"url": final_url, "status": "success", "file_saved_as": json_filepath}
        except IOError as e:
            return {"url": final_url, "status": "error", "detail": f"Failed to write JSON file: {e}"}
    else:
        return {"url": final_url, "status": "skipped", "detail": "Could not extract title."}

@app.post("/process-articles", summary="Process CSV and Scrape Metadata", tags=["Scraping"])
async def process_articles_endpoint(background_tasks: BackgroundTasks):
    """
    Loads the CSV file, visits each article URL to scrape metadata,
    and saves it to a separate JSON file. This is a long-running process
    and will be executed in the background.
    
    This corresponds to **Task 2** from the notebook.
    """
    if not os.path.exists(CSV_FILENAME):
        raise HTTPException(status_code=404, detail=f"CSV file '{CSV_FILENAME}' not found. Please run /scrape-links first.")

    # This function will be run by the background task
    async def run_processing():
        print("--- Task 2: Background processing of articles started ---")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # The client will now automatically follow redirects
        async with httpx.AsyncClient(headers=headers, timeout=20.0, follow_redirects=True) as client: # <--- CHANGE HERE
            with open(CSV_FILENAME, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    _, _, article_url = row
                    print(f"Processing article {i+1}: {article_url}")
                    result = await process_single_article(article_url, client)
                    print(f"  -> Result: {result['status']} - {result.get('detail') or result.get('file_saved_as')}")
        print("--- Background processing finished ---")

    background_tasks.add_task(run_processing)
    
    return {
        "message": "Article processing started in the background.",
        "details": f"Monitor the server console and the '{JSON_OUTPUT_DIR}' directory for progress."
    }

@app.get("/articles", summary="List All Scraped Articles", response_model=List[Dict], tags=["Data Access"])
def get_all_articles():
    """
    Retrieves all scraped article metadata from the JSON files.
    """
    articles = []
    if not os.path.exists(JSON_OUTPUT_DIR):
        return []
        
    for filename in sorted(os.listdir(JSON_OUTPUT_DIR)): # Sort for consistent order
        if filename.endswith('.json'):
            filepath = os.path.join(JSON_OUTPUT_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    articles.append(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Could not read or parse {filename}: {e}")
                continue # Skip corrupted or unreadable files
                
    return articles

# --- Uvicorn Runner (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)