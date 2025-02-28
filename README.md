# G4 Investor

## Overview

The G4 Investor is a financial advisory chatbot designed to provide users with quick and accurate answers to their questions related to public companies listed on NASDAQ. The chatbot leverages advanced technologies in finance and AI, including LangChain, FastAPI, and ChromaDB, to deliver a seamless user experience.

## Project Structure

The project is organized into several components:

- **api**: Contains the FastAPI application, including API endpoints, services, database models, and utility functions.
- **front-chat**: Contains the user interface for interacting with the chatbot.
- **backend-api**: Contains the backend of the application.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your machine.
- A populated ChromaDB stored in a Docker named volume.

## To store a populated ChromaDB in a Docker named volume (from a .zip file)

### Steps

1. Navigate to a folder in your local machine. For example, `test_directory/`.

2. Download the `POPULATED_CHROMA_DB.zip` file, and uncompress it inside `test_directory/`. This will output a folder and a `.sqlite` file (together, they should weigh >5GB). We are going to mount those items into a Docker volume.

3. Create the Docker volume by running

    ```shell
    docker volume create G4-investor_chroma_persist_storage 
    ```

    **Note: the name of the Docker volume has to follow this naming paradigm: `{repository-name}_{name-of-volume-in-docker-compose.yml}`. If you check out the `docker-compose.yml` file in the project's root, you'll see that the volume associated to the ChromaDB container is called `chroma_persist_storage`. That corresponds to the `{name-of-volume-in-docker-compose.yml}`, and the `{repository-name}` would be `G4-investor` in this case.**

4. Copy the uncompressed items into the volume by running a temporary container:

    ```shell
    docker run --rm -v G4-investor_chroma_persist_storage:/mnt/volume -v /test_directory:/mnt/source ubuntu bash -c "cp -r /mnt/source/* /mnt/volume"
    ```
  
    Now, the contents of `test_directory/` live inside the Docker volume. You can safely remove the contents of `test_directory/` if you wish (to save memory space).

## To get the application up and running

### Steps

1. Clone the repository in a local folder of your own choosing:

   ```git
   git clone <repository-url>
   cd G4-investor
   ```

2. Create the `.env` file from `.env.original`. Standing on the project's root, run

    ```shell
    cp .env.original .env
    ```

3. Set the value of the `OPENAI_API_KEY` and e-mail configuration variables inside your newly created `.env` file.

4. Repeat steps 2 and 3 for the `.env.original` files found in the `front-chat` and `backend-api` directories, respectively.

5. You can now run the application with

    ```shell
    docker-compose up --build -d
    ```

6. Access the frontend application at `http://localhost:8501`.

7. The FastAPI backend can be accessed at `http://localhost:8004/docs` for API documentation.

## Usage

- Users can interact with the chatbot through the frontend interface, asking questions related to NASDAQ companies.
- The backend processes these queries, retrieves relevant information from the vector database, and generates responses using LangChain.

## Note: To get the content of ChromaDB to persist across re-deploys

You have to create the `PERSIST_DIRECTORY` environment variable in your `.env` file and set its value to `/chroma/whatever-you-want` (the default path where ChromaDB stores its data is `./chroma`, as stated [here](https://cookbook.chromadb.dev/core/storage-layout/)). That value is going to be the path inside the ChromaDB container where our named volume is going to be mounted. For example, `/chroma/my_db`. Note: you **cannot** set this variable as simply `/chroma/` or any such variation, as Docker will throw an error upon running the containers.

Then, in your `docker-compose.yml` file, set the ChromaDB service as follows:

```docker
chromadb:
    container_name: chromadb
    image: chromadb/chroma:latest
    volumes:
      - index_data:/chroma/.chroma/index
      - chroma_persist_storage:${PERSIST_DIRECTORY}
    ports:
      - "8000:8000"
    env_file:
      - ./.env
```

This will automatically create a named volume called `repository-name_chroma_persist_storage` on your first `up`, and that's where your embeddings are going to live.
