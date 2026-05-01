# Astha Medical Chatbot

Astha is a Flask-based medical question-answering chatbot built with LangChain, Pinecone, OpenAI, and Hugging Face embeddings. It uses a RAG pipeline to retrieve relevant chunks from an indexed medical PDF and generate concise answers through a web chat interface.

## Features

- Medical RAG chatbot with Pinecone vector search
- OpenAI chat model for answer generation
- Hugging Face sentence-transformer embeddings
- Responsive chat UI with Flask templates and static CSS
- Dockerized production runtime with Gunicorn
- GitHub Actions CI/CD to Amazon ECR and EC2
- Self-hosted GitHub runner based deployment on EC2

## Project Structure

```text
.
├── app.py                         # Flask app and RAG endpoint
├── Dockerfile                     # Production Docker image
├── requirements.txt               # Python dependencies
├── data/
│   └── Medical_book.pdf           # Source document for indexing
├── src/
│   ├── helper.py                  # PDF loading, splitting, embeddings
│   ├── prompt.py                  # RAG prompt template
│   └── store_index.py             # Pinecone indexing script
├── static/
│   └── style.css                  # Chat UI styles
├── templates/
│   └── chat.html                  # Chat page
└── .github/workflows/
    └── cicd.yaml                  # ECR build and EC2 deployment
```

## Tech Stack

- Python 3.12
- Flask 3.1
- LangChain 0.3
- Pinecone
- OpenAI API
- Sentence Transformers
- Docker
- Gunicorn
- AWS ECR + EC2
- GitHub Actions

## Environment Variables

Create a `.env` file locally:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

For GitHub Actions deployment, configure these repository secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
OPENAI_API_KEY
PINECONE_API_KEY
```

`ECR_REPO` should be only the ECR repository name, not the full URL.

Example:

```text
ECR_REPO=astha-medical-chatbot
```

## Local Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv myenv
source myenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8080
```

## Build the Pinecone Index

Before the chatbot can answer from your PDF, the Pinecone index must be populated.

Run:

```bash
python src/store_index.py
```

The script:

- loads PDFs from `data/`
- creates text chunks
- embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
- uploads vectors to the Pinecone index:

```text
medical-chatbot-index
```

You only need to rerun this when the source documents or embedding model change.

## API Endpoint

The frontend sends chat messages to:

```text
POST /get
```

Example:

```bash
curl -X POST http://127.0.0.1:8080/get \
  -H "Content-Type: application/json" \
  -d '{"message":"What is acne?"}'
```

Response:

```json
{
  "answer": "Acne is a common skin disease..."
}
```

## Docker

Build the image:

```bash
docker build -t astha-medical-chatbot .
```

Run the container:

```bash
docker run -d \
  --name astha-medical-chatbot \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e PINECONE_API_KEY="$PINECONE_API_KEY" \
  -p 8080:8080 \
  astha-medical-chatbot
```

Open:

```text
http://localhost:8080
```

## AWS Deployment

This project uses GitHub Actions with two jobs:

1. `Continuous-Integration`
   - builds the Docker image
   - pushes it to Amazon ECR

2. `Continuous-Deployment`
   - runs on the EC2 self-hosted GitHub runner
   - pulls the latest image from ECR
   - stops and removes the old container
   - starts the new container on port `8080`

Live app format:

```text
http://YOUR_EC2_PUBLIC_IP:8080
```

> Deployment status note: the previous EC2 instance used for this project was terminated to avoid ongoing AWS charges. To deploy again, create a new EC2 instance, install Docker, configure a new self-hosted GitHub runner, update the security group, and rerun the GitHub Actions workflow.

Make sure the EC2 security group allows:

```text
Custom TCP 8080
SSH 22
```

## Self-Hosted Runner Notes

On EC2, the runner should be installed as a service:

```bash
cd ~/actions-runner
sudo ./svc.sh status
sudo ./svc.sh start
```

Expected status:

```text
active (running)
Connected to GitHub
Listening for Jobs
```

If it is not installed:

```bash
cd ~/actions-runner
sudo ./svc.sh install
sudo ./svc.sh start
```

## Troubleshooting

### `Waiting for a runner to pick up this job`

The EC2 self-hosted runner is offline.

Fix:

```bash
cd ~/actions-runner
sudo ./svc.sh start
sudo ./svc.sh status
```

### `Numpy is not available`

This is usually a NumPy/Torch compatibility issue in Docker. The project pins:

```text
numpy==1.26.4
torch==2.2.2
```

Rebuild and redeploy the Docker image after dependency changes.

### `no space left on device`

Docker layers filled the EC2 disk.

Fix on EC2:

```bash
docker system prune -a -f
docker volume prune -f
df -h
```

### App opens but chat response fails

Check container logs:

```bash
docker logs astha-medical-chatbot --tail 100
```

Also confirm these environment variables are passed into the container:

```text
OPENAI_API_KEY
PINECONE_API_KEY
```

## Production Start Command

The Docker container runs:

```bash
gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:8080
```

## Disclaimer

This chatbot is for educational and informational support only. It is not a replacement for professional medical advice, diagnosis, or treatment.
