# Web scrape policy

- Respect `robots.txt` where practical; skip paywalled or login-only pages.
- One request per URL during ingest; no aggressive crawling.
- Use a descriptive User-Agent (`ResearchMind/0.1`).
- On HTTP errors, surface the status code to the user and do not index empty pages.
