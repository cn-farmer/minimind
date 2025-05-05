chunks = ["Hello", " world", "!"]
history_idx = 0
for chunk in chunks:
    print(chunks[history_idx:] if history_idx < len(chunk) else chunk, end='', flush=True)
    history_idx = len(chunk)  # 更新位置