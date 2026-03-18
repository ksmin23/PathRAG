# PathRAG Web App - API Reference

This document describes the REST API endpoints and data models for the PathRAG web application.

## Base URL

- Development: `http://localhost:8000`
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- API Documentation (ReDoc): `http://localhost:8000/redoc`

## Authentication

All endpoints (except `/token` and `/register`) require a valid JWT token in the `Authorization` header:
```
Authorization: Bearer <access_token>
```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/token` | Login and get access token |
| `POST` | `/register` | Register a new user |
| `GET` | `/users/me` | Get current user information |

### Users

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/users/` | Get all users |
| `POST` | `/users/theme` | Update user theme |

### Chat Threads

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/chats/threads` | Get all chat threads |
| `POST` | `/chats/threads` | Create a new chat thread |
| `GET` | `/chats/threads/{thread_uuid}` | Get a specific thread with all its chats |
| `PUT` | `/chats/threads/{thread_uuid}` | Update a thread's title |
| `DELETE` | `/chats/threads/{thread_uuid}` | Mark a thread as deleted |

### Chats

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/chats/` | Get all chats |
| `GET` | `/chats/recent` | Get the 5 most recent chat threads |
| `POST` | `/chats/chat/{thread_uuid}` | Create a new chat message in a thread |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/documents/` | Get all documents |
| `POST` | `/documents/upload` | Upload a document |
| `GET` | `/documents/{document_id}` | Get a specific document |
| `GET` | `/documents/{document_id}/status` | Get document processing status |
| `POST` | `/documents/reload` | Reload the PathRAG instance to recognize new documents |

### Knowledge Graph

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/knowledge-graph/` | Get the knowledge graph |
| `POST` | `/knowledge-graph/query` | Query the knowledge graph |

## Data Models

### User

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary Key |
| `username` | String | Unique username |
| `email` | String | Unique email |
| `hashed_password` | String | Bcrypt hashed password |
| `created_at` | DateTime | Account creation timestamp |
| `theme` | String | UI theme preference (default: "blue") |

### Thread

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary Key |
| `uuid` | String | Unique thread identifier |
| `user_id` | Integer | Foreign Key to User |
| `title` | String | Thread title (auto-generated from first message) |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |
| `is_deleted` | Boolean | Soft delete flag (default: False) |

### Chat

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary Key |
| `user_id` | Integer | Foreign Key to User |
| `thread_id` | Integer | Foreign Key to Thread |
| `role` | String | Message role ('user' or 'system') |
| `message` | Text | Message content |
| `created_at` | DateTime | Message timestamp |

### Document

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary Key |
| `user_id` | Integer | Foreign Key to User |
| `filename` | String | Original filename |
| `content_type` | String | MIME type |
| `file_path` | String | Server file path |
| `file_size` | Integer | File size in bytes |
| `uploaded_at` | DateTime | Upload timestamp |
| `status` | String | Processing status |
| `processed_at` | DateTime | Processing completion timestamp (nullable) |
| `error_message` | Text | Error details if processing failed (nullable) |

## Default Users

The application creates the following default users on first startup:

| Username | Email | Password |
|----------|-------|----------|
| `user1` | user1@example.com | `Pass@123` |
| `user2` | user2@example.com | `Pass@123` |
| `user3` | user3@example.com | `Pass@123` |
