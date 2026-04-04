from input_classifier import InputClassifier

classifier = InputClassifier()

# 测试当前轮次名字提取
test_cases = [
    # (输入, 历史, 预期类型, 预期结果)
    ("I am Tom. Who am I?", [], "identity", "You are Tom."),
    ("my name is Alice. Who am I?", [], "identity", "You are Alice."),
    ("who am i?", [], "rag", None),  # 无历史 + 当前无名字 → 降级 RAG
    ("who am i?", [{"role": "user", "content": "my name is Bob"}], "identity", "You are Bob."),
]

print("=" * 60)
print("🧪 当前轮次名字提取测试")
print("=" * 60)

for query, history, expected_type, expected_value in test_cases:
    result = classifier.classify(query, history)
    
    type_match = result["type"] == expected_type
    value_match = result["value"] == expected_value if expected_value else True
    status = "✅" if (type_match and value_match) else "❌"
    
    print(f"\n{status} 输入: [{query}]")
    print(f"   预期类型: {expected_type}")
    print(f"   实际类型: {result['type']}")
    print(f"   结果: {result['value']}")
    if "fallback_reason" in result:
        print(f"   降级原因: {result['fallback_reason']}")
