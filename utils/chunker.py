"""文档智能切分模块。

对超过 CHUNK_SIZE 的文档按类型（制表符表格 / 密集清单 / 普通文本）
走不同的切分管线，输出与 doc_parser 相同格式的三元组。
"""
from typing import List, Dict, Tuple


def chunk_documents(
    docs_loc: List[str],
    docs_content: List[str],
    metadatas: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> Tuple[List[str], List[str], List[Dict]]:
    """主入口：遍历所有文档，超限的切分，未超限的透传。"""
    out_loc: List[str] = []
    out_content: List[str] = []
    out_metadatas: List[Dict] = []
    stats = {"passed": 0, "split": 0, "sub_chunks": 0}

    for loc, content, meta in zip(docs_loc, docs_content, metadatas):
        if len(content) <= chunk_size:
            out_loc.append(loc)
            out_content.append(content)
            out_metadatas.append(meta)
            stats["passed"] += 1
            continue

        chunk_type = _detect_chunk_type(content)
        if chunk_type == "tab_table":
            sub_chunks = _split_tab_table(content, chunk_size)
        elif chunk_type == "space_table":
            sub_chunks = _split_space_table(content, chunk_size)
        else:
            sub_chunks = _split_prose(content, chunk_size, chunk_overlap)

        stats["split"] += 1
        stats["sub_chunks"] += len(sub_chunks)

        for seq, sub_content in enumerate(sub_chunks):
            out_content.append(sub_content)
            out_loc.append(f"{loc} [chunk {seq + 1}/{len(sub_chunks)}]")
            out_metadatas.append({
                **meta,
                "content": sub_content,
                "chunk_seq": seq,
                "chunk_total": len(sub_chunks),
            })

    print(f"切分完成：{stats['passed']} 条透传，"
          f"{stats['split']} 条切分为 {stats['sub_chunks']} 个子 chunk，"
          f"输出总数 {len(out_metadatas)} 条")
    return out_loc, out_content, out_metadatas


def _detect_chunk_type(content: str) -> str:
    """检测文档类型：tab_table / space_table / prose。"""
    if '\t' in content:
        return "tab_table"
    first_line = content.split('\n', 1)[0][:200]
    if '序号' in first_line:
        return "space_table"
    return "prose"


def _split_tab_table(content: str, chunk_size: int) -> List[str]:
    """制表符表格切分：按行累积，表头附加到每个子 chunk。"""
    lines = content.split('\n')
    if not lines:
        return [content]

    header_line = lines[0]
    header_len = len(header_line) + 1  # +1 for \n

    # 过滤数据行：跳过空行、续表标记行
    data_lines: List[str] = []
    skip_next_header = False
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # 续表标记行：以"表"开头且含"续"
        if stripped.startswith('表') and '续' in stripped:
            skip_next_header = True
            continue
        # 续表标记后的重复表头行
        if skip_next_header and '\t' in stripped:
            skip_next_header = False
            continue
        skip_next_header = False
        data_lines.append(line)

    if not data_lines:
        return [content]

    # 累积行分组
    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = header_len  # 预留表头空间

    for line in data_lines:
        line_len = len(line) + 1  # +1 for \n
        if current_lines and current_len + line_len > chunk_size:
            chunks.append(header_line + '\n' + '\n'.join(current_lines))
            current_lines = [line]
            current_len = header_len + line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append(header_line + '\n' + '\n'.join(current_lines))

    return chunks if chunks else [content]


def _split_space_table(content: str, chunk_size: int) -> List[str]:
    """密集清单切分：按行累积，表头附加到每个子 chunk。"""
    lines = content.split('\n')
    if not lines:
        return [content]

    header_line = lines[0]
    header_len = len(header_line) + 1

    # 过滤数据行：跳过空行、--- 分隔线
    data_lines: List[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.strip('-').strip() == '' and len(stripped) >= 3:
            continue
        data_lines.append(line)

    if not data_lines:
        return [content]

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = header_len

    for line in data_lines:
        line_len = len(line) + 1
        if current_lines and current_len + line_len > chunk_size:
            chunks.append(header_line + '\n' + '\n'.join(current_lines))
            current_lines = [line]
            current_len = header_len + line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append(header_line + '\n' + '\n'.join(current_lines))

    return chunks if chunks else [content]


# 低语义强度分隔符：仅在这些分隔符切分时启用 overlap
_LOW_SEMANTIC_SEPS = {" "}

_PROSE_SEPARATORS = ["\n\n", "。", "；", "\n", "，", " "]


def _split_prose(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    _sep_idx: int = 0,
) -> List[str]:
    """普通文本递归切分：按优先级分隔符从粗到细依次切分。"""
    if len(content) <= chunk_size:
        return [content]

    # 所有分隔符耗尽，字符级截断
    if _sep_idx >= len(_PROSE_SEPARATORS):
        return _chunk_by_chars(content, chunk_size, chunk_overlap)

    sep = _PROSE_SEPARATORS[_sep_idx]
    fragments = content.split(sep)

    chunks: List[str] = []
    current = ""

    for i, frag in enumerate(fragments):
        # 重新附加分隔符（最后一个片段除外）
        candidate = frag + sep if i < len(fragments) - 1 else frag

        if len(candidate) > chunk_size:
            # 超限片段：先保存当前累积，再递归切分
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                _split_prose(candidate, chunk_size, chunk_overlap, _sep_idx + 1))
        elif not current:
            current = candidate
        elif len(current) + len(candidate) <= chunk_size:
            current += candidate
        else:
            chunks.append(current)
            current = candidate

    if current:
        chunks.append(current)

    # 仅在低语义强度分隔符时添加 overlap
    if sep in _LOW_SEMANTIC_SEPS and chunk_overlap > 0:
        chunks = _apply_overlap(chunks, chunk_overlap)

    return chunks if chunks else [content]


def _chunk_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """兜底：字符级截断。"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap if chunk_overlap > 0 else end
    return chunks


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """在相邻 chunk 之间添加重叠（取前一个 chunk 的尾部拼到后一个 chunk 的头部）。"""
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        result.append(tail + chunks[i])
    return result
