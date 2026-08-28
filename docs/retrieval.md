# Retrieval developer note

An embedding is a numeric representation of text in which semantically similar
text tends to be near other text. Semantic search can therefore match a
paraphrase such as "how long can I send this back?" to a section that says
"return within 30 calendar days", even when the exact keywords differ.

Qdrant stores the embedding vectors and searchable points. The vector data is
used for similarity search; payload metadata is the attached structured data,
including the chunk content, source filename, heading, and frontmatter. The
similarity score only measures closeness in embedding space. It does not decide
whether a document is active, authoritative, customer-facing, or superseded,
so those metadata fields must be considered by a later retrieval stage.

This project sends embeddings through LangChain's native Gemini adapter to the
Gemini API. Configure `GEMINI_API_KEY` and, optionally,
`GEMINI_EMBEDDING_MODEL`; no OpenAI key is required.