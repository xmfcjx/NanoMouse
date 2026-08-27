"""
CS336 风格的 PDF 文本清洗模块

参考斯坦福 CS336 课程中关于网页数据清洗的核心思想：
1. 数据质量 > 模型大小
2. 清洗策略：长度过滤、符号比例过滤、重复词过滤
"""
import re


def clean_pdf_text_336_style(text: str) -> str:
    """
    CS336 风格的 PDF 文本清洗
    
    处理步骤：
    1. 行级清洗：过滤噪声行（页码、页眉、公式、图表编号等）
    2. 保护"假句号"：防止小数点、缩写、代码中的点被误判为句末
    
    :param text: 原始 PDF 提取的文本
    :return: 清洗后的文本（含占位符保护的点号）
    """
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if not _is_valid_line(line):
            continue
        
        cleaned_lines.append(line)
    
    text = ' '.join(cleaned_lines)
    text = _protect_fake_periods(text)
    
    return text


def _is_valid_line(line: str) -> bool:
    """
    判断一行是否为有效内容行
    
    过滤规则（参考 CS336）：
    1. 长度过滤：过短的行通常是噪声
    2. 符号比例过滤：符号占比过高的行通常是公式/图表编号
    3. 重复词过滤：重复词过多的行质量低
    4. 特殊模式过滤：图表编号、参考文献等
    """
    
    if len(line) < 15:
        return False
    
    alpha_count = sum(1 for c in line if c.isalpha())
    if alpha_count / len(line) < 0.4:
        return False
    
    words = line.lower().split()
    if len(words) >= 4:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return False
    
    noise_patterns = [
        r'^(Figure|Table|Fig\.?|Eq\.?|Equation|Section|Chapter|Slide)\s*\d*\.?\d*',
        r'^\[\d+\]',
        r'^\d+$',
        r'^Page\s+\d+',
        r'^\d+\.\d+\s*$',
    ]
    for pattern in noise_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return False
    
    return True


def _protect_fake_periods(text: str) -> str:
    """
    保护"假句号"，防止被正则错误切分
    
    处理以下情况：
    - 小数点：1.5, 0.001, 3.14
    - 缩写：e.g., i.e., etc., vs., Dr., Prof.
    - 代码：nn.Linear(), .to(), .cuda(), torch.randn()
    - URL：http://, https://
    """
    
    text = re.sub(r'(\d)\.(\d)', r'\1__DECIMAL__\2', text)
    
    multi_dot_abbrs = ['i.e', 'e.g']
    for abbr in multi_dot_abbrs:
        escaped = re.escape(abbr)
        text = re.sub(rf'\b{escaped}\.', f'{abbr.replace(".", "__ABBR__")}__ABBR__', text, flags=re.IGNORECASE)
    
    abbreviations = [
        'etc', 'vs', 'Dr', 'Prof', 'Mr', 'Mrs', 'Ms',
        'Fig', 'Eq', 'Sec', 'Chap', 'Vol', 'No', 'pp', 'cf', 'al', 'et'
    ]
    for abbr in abbreviations:
        escaped = re.escape(abbr)
        text = re.sub(rf'\b{escaped}\.(?=\s|,|;|$)', f'{abbr}__ABBR__', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(\w)\.(\w+\()', r'\1__CODE__\2', text)
    text = re.sub(r'\.(\w+)\(', r'__CODE__\1(', text)
    
    def protect_url(match):
        url = match.group(0)
        url = url.replace(':', '__URL__')
        url = url.replace('.', '__URLDOT__')
        return url
    text = re.sub(r'https?://[^\s]+', protect_url, text)
    
    return text


def restore_protected_periods(text: str) -> str:
    """
    恢复被保护的点号
    
    在 split_text 切分完成后调用，将占位符还原为真实的点号
    """
    text = text.replace('__URLDOT__', '.')
    text = text.replace('__DECIMAL__', '.')
    text = text.replace('__ABBR__', '.')
    text = text.replace('__CODE__', '.')
    text = text.replace('__URL__', ':')
    return text
