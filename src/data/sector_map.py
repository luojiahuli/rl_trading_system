#!/usr/bin/env python3
"""板块映射表 - 新闻关键词 → 股票板块"""
import re

# 关键词 → 板块映射
SECTOR_KEYWORDS = {
    "人工智能": ["人工智能", "AI", "大模型", "机器学习", "深度学习", "智能体", "算力"],
    "半导体": ["半导体", "芯片", "集成电路", "光刻", "封测", "晶圆", "EDA"],
    "新能源": ["新能源", "锂电池", "光伏", "风电", "氢能", "储能", "电池"],
    "新能源汽车": ["新能源汽车", "电动汽车", "充电桩", "锂电", "比亚迪", "特斯拉"],
    "数字经济": ["数字经济", "数据要素", "数据资产", "数字化转型", "云计算", "大数据"],
    "国产软件": ["国产软件", "信创", "操作系统", "数据库", "工业软件", "鸿蒙"],
    "军工": ["军工", "国防", "航天", "航空", "舰船", "导弹", "卫星"],
    "医药": ["医药", "医疗", "创新药", "CXO", "生物医药", "医疗器械"],
    "消费电子": ["消费电子", "智能手机", "可穿戴", "MR", "VR", "AR", "折叠屏"],
    "机器人": ["机器人", "人形机器人", "工业机器人", "伺服", "减速器"],
    "算力": ["算力", "服务器", "GPU", "光模块", "交换机", "数据中心", "CPO"],
    "通信": ["通信", "5G", "6G", "光通信", "卫星互联网"],
    "光伏": ["光伏", "太阳能", "硅料", "硅片", "组件", "逆变器"],
    "储能": ["储能", "抽水蓄能", "电化学储能", "钠离子", "钒电池"],
    "低空经济": ["低空经济", "无人机", "eVTOL", "飞行汽车", "空管"],
    "中特估": ["中特估", "央企改革", "国企改革", "中字头", "一带一路"],
    "消费": ["消费", "食品饮料", "白酒", "家电", "旅游", "免税"],
    "教育": ["教育", "职业教育", "AI教育", "在线教育", "高教"],
    "农业": ["农业", "种业", "粮食", "农机", "猪周期", "乡村振兴"],
    "环保": ["环保", "碳中和", "碳交易", "节能减排", "污水处理"],
    "基建": ["基建", "水利", "轨道交通", "公路", "工程机械"],
}

# 排除的板块关键词
EXCLUDED_KEYWORDS = [
    "银行", "保险", "证券", "信托", "金融", "券商", "房地产",
    "多元金融", "参股银行",
]


def map_keywords_to_sectors(keywords: list) -> list:
    """将关键词列表映射到板块，返回 [(sector, match_count), ...]"""
    matches = {}
    for word in keywords:
        word_lower = word.lower()
        for sector, sector_kws in SECTOR_KEYWORDS.items():
            for kw in sector_kws:
                if kw.lower() in word_lower:
                    matches[sector] = matches.get(sector, 0) + 1
                    break
    # 排除金融券商
    for ex in EXCLUDED_KEYWORDS:
        matches.pop(ex, None)
    sorted_matches = sorted(matches.items(), key=lambda x: -x[1])
    return sorted_matches


def extract_hot_sectors_from_news(news_items: list) -> list:
    """从新闻列表中提取热门板块"""
    all_text = " ".join([item.get("title", "") + " " + item.get("content", "")
                         for item in news_items])
    import jieba
    words = jieba.lcut(all_text)
    # 过滤停用词和短词
    words = [w for w in words if len(w) > 1]
    return map_keywords_to_sectors(words)


def get_sector_stock_codes(sector_name: str, max_count: int = 10) -> list:
    """获取板块对应的股票代码（排除金融券商）"""
    import akshare as ak
    try:
        df = ak.stock_board_industry_cons_em(symbol=sector_name)
        if "代码" not in df.columns:
            return []
        # 简单排除金融股（6xxxxx 银行保险等通过名称过滤）
        codes = df["代码"].tolist()
        names = df.get("名称", df.get("股票名称", [])).tolist()
        result = []
        for code, name in zip(codes, names):
            name_str = str(name or "")
            if not any(ex in name_str for ex in EXCLUDED_KEYWORDS):
                result.append(code)
            if len(result) >= max_count:
                break
        return result
    except Exception:
        return []
