# BBC Sports Web Scraper API

This project provides a FastAPI-based web service to scrape sports articles from the BBC. It's a conversion of a Jupyter Notebook script into a modular and deployable API.

## Features

- **Scrape Links**: An endpoint to fetch all current article links from the BBC Sports homepage and save them to a CSV file.
- **Process Articles**: An endpoint to read the CSV, visit each link, scrape detailed metadata (title, summary, content, etc.), and save each article as a separate JSON file.
- **Retrieve Data**: An endpoint to list all the scraped article data.

## Project Structure

```
/fastapi_webscrap/
├── main.py              # The FastAPI application code
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Setup and Installation

1.  **Clone the repository or create the files** as shown above.

2.  **Create a virtual environment** (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

To start the API server, run the following command in your terminal from the project's root directory:

```sh
uvicorn main:app --reload
```

-   `uvicorn`: The ASGI server.
-   `main:app`: Tells uvicorn to look for an object named `app` in the file `main.py`.
-   `--reload`: Automatically restarts the server whenever you make changes to the code.

Once running, the API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## API Usage

The recommended workflow is to first call `/scrape-links` and then `/process-articles`.

### 1. `POST /scrape-links`

This endpoint triggers the scraping of the BBC Sports homepage to find article URLs. It saves the links to `bbc_sports_articles.csv`.

**Example using `curl`:**
```sh
curl -X POST http://127.0.0.1:8000/scrape-links
```

**Example Success Response:**
```json
{
  "message": "Successfully saved 38 article links.",
  "file_saved_as": "bbc_sports_articles.csv",
  "links_found": 38
}
```

### 2. `POST /process-articles`

This endpoint reads the `bbc_sports_articles.csv` file and starts a background task to scrape each article's metadata. The metadata is saved into individual JSON files in the `article_metadata/` directory.

**Example using `curl`:**
```sh
curl -X POST http://127.0.0.1:8000/process-articles
```

**Example Success Response:**
```json
{
  "message": "Article processing started in the background.",
  "details": "Moniter the server console and the 'article_metadata' directory for progress."
}
```
You can monitor the server's console output to see the progress of the background task.

### 3. `GET /articles`

This endpoint retrieves and returns all the data from the JSON files created by the `/process-articles` endpoint.

**Example using `curl`:**
```sh
curl -X GET http://127.0.0.1:8000/articles
```

**Example Success Response:**
```json
[
  {
    "url": "https://www.bbc.com/sport/football/69876543",
    "title": "Example Football Match Report",
    "summary": "A thrilling match ended in a draw...",
    "publish_date": "2024-07-08T12:00:00.000Z",
    "article_image": "https://ichef.bbci.co.uk/images/ic/1024x576/p0hjs123.jpg",
    "article_content": "The first half was a tense affair..."
  },
  {
    "url": "https://www.bbc.com/sport/tennis/69871234",
    "title": "Tennis Star Advances to Final",
    "summary": "A dominant performance from the world number one...",
    "publish_date": "2024-07-08T11:30:00.000Z",
    "article_image": "https://ichef.bbci.co.uk/images/ic/1024x576/p0hkf456.jpg",
    "article_content": "From the first serve, it was clear who was in control..."
  }
]
```
