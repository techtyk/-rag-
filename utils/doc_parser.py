import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


# ------------------------------
# 解析法规 JSON（不依赖 type，仅根据 children 和 content）
# ------------------------------

def parse_regulation(path: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    支持单 JSON 文件或目录。
    返回：
        docs_index_loc: 定位索引文本列表（文件名 + 章节 + 条款号）
        docs_index_content: 内容索引文本列表（仅正文）
        metadatas: 元数据列表（包含 file_name, chapter, article_no, content）
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    # 收集所有 JSON 文件
    if path_obj.is_file() and path_obj.suffix.lower() == '.json':
        json_files = [path_obj]
    elif path_obj.is_dir():
        json_files = list(path_obj.glob("*.json"))
    else:
        raise ValueError(f"路径不是 JSON 文件或目录: {path}")

    if not json_files:
        raise ValueError(f"未找到任何 JSON 文件: {path}")
    
    docs_index_loc = []
    docs_index_content = []
    metadatas = []

    # 递归解析函数
    def traverse(node, current_chapter: str, file_name: str):
        title = node.get("title", "").strip()
        content = node.get("content", "").strip()
        has_children = bool(node.get("children"))

        # 章节节点
        if content and has_children:
            chapter_name = title if title else content
            current_chapter = chapter_name

        # 条款节点
        if content and not has_children:
            article_no = title if title else ""
            # 定位索引文本：文件名 + 当前章节 + 条款号
            loc_text = f"{file_name} {current_chapter} {article_no}"
            docs_index_loc.append(loc_text)
            # 内容索引文本：仅正文
            docs_index_content.append(content)
            metadatas.append({
                "file_name": file_name,
                "chapter": current_chapter,
                "article_no": article_no,
                "content": content
            })

        for child in node.get("children", []):
            traverse(child, current_chapter, file_name)

    # 遍历每个 JSON 文件
    for json_file in json_files:
        file_name = json_file.stem   # 不含扩展名的文件名，如 "公安机关办理刑事案件程序规定(2020)"
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从 structure 根节点开始遍历，初始 current_chapter 为空
        for root in data.get("structure", []):
            traverse(root, "", file_name)

    print(f"共处理 {len(json_files)} 个法规文件，提取 {len(metadatas)} 个条款")
    return docs_index_loc, docs_index_content, metadatas


