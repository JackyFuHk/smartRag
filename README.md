# smartRag
The system leverages Azure OpenAI as the base model, BGE for embeddings, and Qdrant as the vector database. Built with LangChain, it forms a robust RAG (Retrieval-Augmented Generation) framework capable of processing PDFs, images, and tabular datasets.

The system is designed to be deployed on Azure, and can be easily integrated with other applications and services. The system can be used to search and retrieve relevant documents from a large collection of documents, and then generate new documents based on the retrieved documents. The system can be used for a variety of applications, such as document retrieval, document generation, and knowledge base building.

## 20250802 Updates
Complete the file upload process, extract its contents, generate embeddings, and store them in Qdrant. The system can now search and retrieve relevant documents from a large collection of documents. with metadata like below:
'''
{
    "id": "1234567890",
    "title": "Document Title",
    "content": "Document Content",
    "url": "https://example.com/document",
    "created_at": "2021-01-01T00:00:00Z",
    "updated_at": "2021-01-01T00:00:00Z"
    "user_id":"admin"
}
'''

