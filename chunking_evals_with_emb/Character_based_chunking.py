def char_chunk_text(text: str, chunk_size: int, overlap: int = 0):
    """
    Fixed-length character chunking with overlap.

    chunk_size: number of characters per chunk
    overlap:    number of characters repeated from previous chunk
    """
    assert chunk_size > 0, "chunk_size must be > 0"
    assert 0 <= overlap < chunk_size, "overlap must be in [0, chunk_size)"

    chunks = []
    stride = chunk_size - overlap
    i = 0

    while i < len(text):
        chunk = text[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        i += stride

    return chunks
