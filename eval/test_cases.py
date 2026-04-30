"""
测试用例定义
"""
import random

RAG_TEST_CASES = [
    {"id": 1, "question": "What is QuickSort?", "expected_keywords": ["divide-and-conquer", "partitioning", "O(n log n)"]},
    {"id": 2, "question": "How does BERT work?", "expected_keywords": ["bidirectional", "pre-training", "masked language modeling"]},
    {"id": 3, "question": "What is the difference between TCP and UDP?", "expected_keywords": ["reliable", "connection-oriented", "datagram"]},
    {"id": 4, "question": "What is virtual memory?", "expected_keywords": ["page tables", "demand paging", "address space"]},
    {"id": 5, "question": "How does a GPU differ from a CPU?", "expected_keywords": ["SIMD", "parallelism", "thousands of cores"]},
    {"id": 6, "question": "What is the CAP theorem?", "expected_keywords": ["consistency", "availability", "partition tolerance"]},
    {"id": 7, "question": "What is Moore's Law?", "expected_keywords": ["transistor", "doubling", "two years"]},
    {"id": 8, "question": "How does TLS work?", "expected_keywords": ["certificate", "session key", "authentication"]},
    {"id": 9, "question": "What is dynamic programming?", "expected_keywords": ["overlapping subproblems", "memoization", "tabulation"]},
    {"id": 10, "question": "What are design patterns?", "expected_keywords": ["creational", "structural", "behavioral"]},
    {"id": 11, "question": "What is the Von Neumann architecture?", "expected_keywords": ["stored-program", "memory", "instructions"]},
    {"id": 12, "question": "How does pipelining work in CPU?", "expected_keywords": ["fetch", "decode", "execute", "hazard"]},
    {"id": 13, "question": "What is a hash table?", "expected_keywords": ["O(1)", "collision", "chaining"]},
    {"id": 14, "question": "What is the difference between SVM and decision tree?", "expected_keywords": ["hyperplane", "margin", "information gain"]},
    {"id": 15, "question": "What is CMOS technology?", "expected_keywords": ["complementary", "n-type", "p-type", "static power"]},
    {"id": 16, "question": "What is out-of-order execution?", "expected_keywords": ["Tomasulo", "reservation stations", "dynamic"]},
    {"id": 17, "question": "How does Ethernet work?", "expected_keywords": ["CSMA/CD", "MAC", "frame"]},
    {"id": 18, "question": "What is a deadlock?", "expected_keywords": ["mutual exclusion", "circular wait", "resource"]},
    {"id": 19, "question": "What is gradient boosting?", "expected_keywords": ["iteratively", "correct errors", "ensemble"]},
    {"id": 20, "question": "What is the difference between RISC and CISC?", "expected_keywords": ["fixed-length", "variable-length", "single cycle"]},
]


def generate_agent_test_cases(n=500):
    cases = []
    case_id = 1
    
    cities = [
        "Beijing", "Shanghai", "Tokyo", "New York", "London", "Paris", "Berlin", 
        "Sydney", "Moscow", "Dubai", "Singapore", "Hong Kong", "Seoul", "Mumbai",
        "Toronto", "Chicago", "Los Angeles", "San Francisco", "Seattle", "Boston",
        "Miami", "Denver", "Phoenix", "Houston", "Dallas", "Atlanta", "Philadelphia",
        "Rome", "Madrid", "Barcelona", "Amsterdam", "Vienna", "Prague", "Warsaw",
        "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Dublin", "Brussels", "Lisbon"
    ]
    
    while len(cases) < n:
        if len(cases) < n:
            a, b = random.randint(1, 100), random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Calculate {a} + {b}",
                "tool": "calc", "input": f"print({a} + {b})",
                "check_type": "exact", "expected": str(a + b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(50, 200), random.randint(1, 50)
            cases.append({
                "id": case_id, "question": f"Calculate {a} - {b}",
                "tool": "calc", "input": f"print({a} - {b})",
                "check_type": "exact", "expected": str(a - b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(2, 20), random.randint(2, 20)
            cases.append({
                "id": case_id, "question": f"Calculate {a} * {b}",
                "tool": "calc", "input": f"print({a} * {b})",
                "check_type": "exact", "expected": str(a * b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(10, 200), random.randint(2, 10)
            if a % b == 0:
                cases.append({
                    "id": case_id, "question": f"Calculate {a} / {b}",
                    "tool": "calc", "input": f"print({a} / {b})",
                    "check_type": "contains", "expected": str(a // b)
                })
                case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(2, 15), random.randint(2, 6)
            cases.append({
                "id": case_id, "question": f"Calculate {a} ** {b}",
                "tool": "calc", "input": f"print({a} ** {b})",
                "check_type": "exact", "expected": str(a ** b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(10, 100), random.randint(2, 10)
            cases.append({
                "id": case_id, "question": f"Calculate {a} % {b}",
                "tool": "calc", "input": f"print({a} % {b})",
                "check_type": "exact", "expected": str(a % b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 5)
            cases.append({
                "id": case_id, "question": f"Calculate ({a} + {b}) * {c}",
                "tool": "calc", "input": f"print(({a} + {b}) * {c})",
                "check_type": "exact", "expected": str((a + b) * c)
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(2, 20)
            cases.append({
                "id": case_id, "question": f"Calculate sqrt({a*a})",
                "tool": "calc", "input": f"print({a*a} ** 0.5)",
                "check_type": "exact", "expected": str(float(a))
            })
            case_id += 1
        
        if len(cases) < n:
            city = random.choice(cities)
            cases.append({
                "id": case_id, "question": f"What is the weather in {city}?",
                "tool": "weather", "input": city,
                "check_type": "weather"
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Solve {a}x = {a * random.randint(2, 20)}",
                "tool": "solve", "input": f"{a}*x={a * random.randint(2, 20)}",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(1, 20), random.randint(1, 50)
            cases.append({
                "id": case_id, "question": f"Solve x + {a} = {a + b}",
                "tool": "solve", "input": f"x+{a}={a+b}",
                "check_type": "contains", "expected": str(b)
            })
            case_id += 1
        
        if len(cases) < n:
            a, b = random.randint(2, 10), random.randint(1, 20)
            cases.append({
                "id": case_id, "question": f"Solve {a}x - {b} = {a * random.randint(3, 15) - b}",
                "tool": "solve", "input": f"{a}*x-{b}={a * random.randint(3, 15) - b}",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(2, 15)
            cases.append({
                "id": case_id, "question": f"Solve x * x = {a * a}",
                "tool": "solve", "input": f"x*x={a*a}",
                "check_type": "contains", "expected": str(a)
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(10, 255)
            cases.append({
                "id": case_id, "question": f"Convert {a} to binary",
                "tool": "base_convert", "input": f"{a} to bin",
                "check_type": "exact", "expected": bin(a)[2:]
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(10, 255)
            cases.append({
                "id": case_id, "question": f"Convert {a} to hexadecimal",
                "tool": "base_convert", "input": f"{a} to hex",
                "check_type": "exact", "expected": hex(a)[2:].upper()
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(10, 255)
            cases.append({
                "id": case_id, "question": f"Convert {a} to octal",
                "tool": "base_convert", "input": f"{a} to oct",
                "check_type": "exact", "expected": oct(a)[2:]
            })
            case_id += 1
        
        if len(cases) < n:
            bits = random.randint(4, 8)
            a = random.randint(0, 2**bits - 1)
            cases.append({
                "id": case_id, "question": f"Convert {bin(a)[2:]} binary to decimal",
                "tool": "base_convert", "input": f"{bin(a)[2:]} to dec",
                "check_type": "exact", "expected": str(a)
            })
            case_id += 1
        
        if len(cases) < n:
            a = random.randint(10, 255)
            hex_val = hex(a)[2:].upper()
            cases.append({
                "id": case_id, "question": f"Convert {hex_val} hex to decimal",
                "tool": "base_convert", "input": f"{hex_val} to dec",
                "check_type": "exact", "expected": str(a)
            })
            case_id += 1
        
        if len(cases) < n:
            km = random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Convert {km}km to miles",
                "tool": "convert", "input": f"{km}km to miles",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            miles = random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Convert {miles} miles to km",
                "tool": "convert", "input": f"{miles} miles to km",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            c = random.randint(-20, 40)
            cases.append({
                "id": case_id, "question": f"Convert {c} celsius to fahrenheit",
                "tool": "convert", "input": f"{c}c to f",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            f = random.randint(0, 120)
            cases.append({
                "id": case_id, "question": f"Convert {f} fahrenheit to celsius",
                "tool": "convert", "input": f"{f}f to c",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            m = random.randint(100, 10000)
            cases.append({
                "id": case_id, "question": f"Convert {m} meters to km",
                "tool": "convert", "input": f"{m} meters to km",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            kg = random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Convert {kg} kg to pounds",
                "tool": "convert", "input": f"{kg} kg to pounds",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            cases.append({
                "id": case_id, "question": "What time is it now?",
                "tool": "time", "input": "",
                "check_type": "time_format"
            })
            case_id += 1
        
        if len(cases) < n:
            cases.append({
                "id": case_id, "question": "What day of the week is today?",
                "tool": "weekday", "input": "",
                "check_type": "weekday"
            })
            case_id += 1
        
        if len(cases) < n:
            year = random.randint(2025, 2030)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            cases.append({
                "id": case_id, "question": f"What weekday is {year}-{month:02d}-{day:02d}?",
                "tool": "weekday", "input": f"{year}-{month:02d}-{day:02d}",
                "check_type": "weekday"
            })
            case_id += 1
        
        if len(cases) < n:
            year = random.randint(2025, 2030)
            month1, day1 = random.randint(1, 12), random.randint(1, 28)
            month2, day2 = random.randint(1, 12), random.randint(1, 28)
            cases.append({
                "id": case_id, "question": f"How many days between {year}-{month1:02d}-{day1:02d} and {year}-{month2:02d}-{day2:02d}?",
                "tool": "days_between", "input": f"{year}-{month1:02d}-{day1:02d}, {year}-{month2:02d}-{day2:02d}",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            year = random.randint(2025, 2030)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            delta = random.randint(1, 100)
            cases.append({
                "id": case_id, "question": f"Calculate {year}-{month:02d}-{day:02d} + {delta} days",
                "tool": "date", "input": f"{year}-{month:02d}-{day:02d} + {delta} days",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            strings = ["hello", "world", "python", "coding", "machine learning", 
                      "artificial intelligence", "deep learning", "neural network",
                      "data science", "computer vision", "natural language"]
            s = random.choice(strings)
            cases.append({
                "id": case_id, "question": f"What is the length of '{s}'?",
                "tool": "str_tools", "input": f"{s}, len",
                "check_type": "exact", "expected": str(len(s))
            })
            case_id += 1
        
        if len(cases) < n:
            strings = ["hello", "world", "python", "coding", "machine"]
            s = random.choice(strings)
            cases.append({
                "id": case_id, "question": f"Reverse the string '{s}'",
                "tool": "str_tools", "input": f"{s}, reverse",
                "check_type": "exact", "expected": s[::-1]
            })
            case_id += 1
        
        if len(cases) < n:
            strings = ["hello world", "python coding", "machine learning", "data science"]
            s = random.choice(strings)
            cases.append({
                "id": case_id, "question": f"Convert '{s}' to uppercase",
                "tool": "str_tools", "input": f"{s}, upper",
                "check_type": "exact", "expected": s.upper()
            })
            case_id += 1
        
        if len(cases) < n:
            strings = ["HELLO WORLD", "PYTHON CODING", "MACHINE LEARNING"]
            s = random.choice(strings)
            cases.append({
                "id": case_id, "question": f"Convert '{s}' to lowercase",
                "tool": "str_tools", "input": f"{s}, lower",
                "check_type": "exact", "expected": s.lower()
            })
            case_id += 1
        
        if len(cases) < n:
            nums = [random.randint(1, 100) for _ in range(random.randint(4, 8))]
            cases.append({
                "id": case_id, "question": f"What is the max of {', '.join(map(str, nums))}?",
                "tool": "statistics", "input": f"{','.join(map(str, nums))}, max",
                "check_type": "exact", "expected": str(max(nums))
            })
            case_id += 1
        
        if len(cases) < n:
            nums = [random.randint(1, 100) for _ in range(random.randint(4, 8))]
            cases.append({
                "id": case_id, "question": f"What is the min of {', '.join(map(str, nums))}?",
                "tool": "statistics", "input": f"{','.join(map(str, nums))}, min",
                "check_type": "exact", "expected": str(min(nums))
            })
            case_id += 1
        
        if len(cases) < n:
            nums = [random.randint(1, 50) for _ in range(random.randint(3, 6))]
            cases.append({
                "id": case_id, "question": f"What is the sum of {', '.join(map(str, nums))}?",
                "tool": "statistics", "input": f"{','.join(map(str, nums))}, sum",
                "check_type": "exact", "expected": str(sum(nums))
            })
            case_id += 1
        
        if len(cases) < n:
            nums = [random.randint(1, 50) for _ in range(random.randint(3, 6))]
            avg = sum(nums) / len(nums)
            cases.append({
                "id": case_id, "question": f"What is the average of {', '.join(map(str, nums))}?",
                "tool": "statistics", "input": f"{','.join(map(str, nums))}, avg",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            n_dice = random.randint(1, 5)
            cases.append({
                "id": case_id, "question": f"Roll a dice {n_dice} time{'s' if n_dice > 1 else ''}",
                "tool": "random", "input": f"dice {n_dice}",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
        
        if len(cases) < n:
            n_coin = random.randint(1, 5)
            cases.append({
                "id": case_id, "question": f"Flip a coin {n_coin} time{'s' if n_coin > 1 else ''}",
                "tool": "random", "input": f"coin {n_coin}",
                "check_type": "contains", "expected": ""
            })
            case_id += 1
    
    return cases[:n]


AGENT_TEST_CASES = generate_agent_test_cases(500)


ROUTING_TEST_CASES = [
    {"question": "What time is it now?", "expected_type": "agent", "expected_subtype": "time"},
    {"question": "What day is today?", "expected_type": "agent", "expected_subtype": "date_today"},
    {"question": "What weekday is 2025-07-05?", "expected_type": "agent", "expected_subtype": "weekday"},
    {"question": "How many days between 2025-01-01 and 2025-06-01?", "expected_type": "agent", "expected_subtype": "days_between"},
    {"question": "What's the weather in Shanghai?", "expected_type": "agent", "expected_subtype": "weather"},
    {"question": "Calculate 15 + 27", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "Solve 3x = 12", "expected_type": "agent", "expected_subtype": "equation"},
    {"question": "Convert 255 to hex", "expected_type": "agent", "expected_subtype": "base_convert"},
    {"question": "Convert 100km to miles", "expected_type": "agent", "expected_subtype": "convert"},
    {"question": "Calculate 5 * 8 then convert to binary", "expected_type": "agent", "expected_subtype": "multi_tool"},
    {"question": "What is machine learning?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does TCP work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is the Von Neumann architecture?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain gradient boosting", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is Moore's Law?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "My name is Alice", "expected_type": "identity", "expected_subtype": "set_name"},
    {"question": "What is my name?", "expected_type": "identity", "expected_subtype": "question"},
    {"question": "Who are you?", "expected_type": "identity", "expected_subtype": "bot"},
    {"question": "I like music", "expected_type": "preference", "expected_subtype": "set"},
    {"question": "What do I like?", "expected_type": "preference", "expected_subtype": "get"},
    {"question": "What's the date today?", "expected_type": "agent", "expected_subtype": "date_today"},
    {"question": "What is 3 * 7?", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "Solve x*x = 16", "expected_type": "agent", "expected_subtype": "equation"},
    {"question": "Convert 1010 to decimal", "expected_type": "agent", "expected_subtype": "base_convert"},
    {"question": "50 fahrenheit to celsius", "expected_type": "agent", "expected_subtype": "convert"},
    {"question": "What is a hash table?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain the CAP theorem", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does virtual memory work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is out-of-order execution?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is CMOS?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Current time please", "expected_type": "agent", "expected_subtype": "time"},
    {"question": "Tell me the time", "expected_type": "agent", "expected_subtype": "time"},
    {"question": "What's the date?", "expected_type": "agent", "expected_subtype": "date_today"},
    {"question": "Today's date", "expected_type": "agent", "expected_subtype": "date_today"},
    {"question": "What day of week is 2025-12-25?", "expected_type": "agent", "expected_subtype": "weekday"},
    {"question": "Days between Jan 1 and Dec 31", "expected_type": "agent", "expected_subtype": "days_between"},
    {"question": "Weather in Beijing", "expected_type": "agent", "expected_subtype": "weather"},
    {"question": "What's the weather like in Tokyo?", "expected_type": "agent", "expected_subtype": "weather"},
    {"question": "Temperature in New York", "expected_type": "agent", "expected_subtype": "weather"},
    {"question": "Calculate 100 / 5", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "What is 25 * 4?", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "Add 15 and 27", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "Subtract 10 from 50", "expected_type": "agent", "expected_subtype": "arithmetic"},
    {"question": "Solve 2x + 5 = 15", "expected_type": "agent", "expected_subtype": "equation"},
    {"question": "Find x when x squared equals 25", "expected_type": "agent", "expected_subtype": "equation"},
    {"question": "Solve for x: 4x = 20", "expected_type": "agent", "expected_subtype": "equation"},
    {"question": "Convert 16 to binary", "expected_type": "agent", "expected_subtype": "base_convert"},
    {"question": "What is FF in decimal?", "expected_type": "agent", "expected_subtype": "base_convert"},
    {"question": "Binary 1111 to decimal", "expected_type": "agent", "expected_subtype": "base_convert"},
    {"question": "50 celsius to fahrenheit", "expected_type": "agent", "expected_subtype": "convert"},
    {"question": "Convert 5 km to meters", "expected_type": "agent", "expected_subtype": "convert"},
    {"question": "100 pounds to kg", "expected_type": "agent", "expected_subtype": "convert"},
    {"question": "What is deep learning?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain neural networks", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is a database?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does HTTP work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is SQL injection?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain REST API", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is Docker?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does Kubernetes work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is microservices?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain CI/CD pipeline", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is blockchain?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "I'm Bob", "expected_type": "identity", "expected_subtype": "set_name"},
    {"question": "Call me John", "expected_type": "identity", "expected_subtype": "set_name"},
    {"question": "My name's Sarah", "expected_type": "identity", "expected_subtype": "set_name"},
    {"question": "Do you know my name?", "expected_type": "identity", "expected_subtype": "question"},
    {"question": "What's my name again?", "expected_type": "identity", "expected_subtype": "question"},
    {"question": "Who am I?", "expected_type": "identity", "expected_subtype": "question"},
    {"question": "What are you?", "expected_type": "identity", "expected_subtype": "bot"},
    {"question": "Tell me about yourself", "expected_type": "identity", "expected_subtype": "bot"},
    {"question": "I enjoy reading", "expected_type": "preference", "expected_subtype": "set"},
    {"question": "I love pizza", "expected_type": "preference", "expected_subtype": "set"},
    {"question": "My favorite color is blue", "expected_type": "preference", "expected_subtype": "set"},
    {"question": "What are my preferences?", "expected_type": "preference", "expected_subtype": "get"},
    {"question": "What do you know about me?", "expected_type": "preference", "expected_subtype": "get"},
    {"question": "What's my favorite?", "expected_type": "preference", "expected_subtype": "get"},
    {"question": "Roll a dice", "expected_type": "agent", "expected_subtype": "random"},
    {"question": "Flip a coin", "expected_type": "agent", "expected_subtype": "random"},
    {"question": "Random number between 1 and 100", "expected_type": "agent", "expected_subtype": "random"},
    {"question": "Length of 'hello world'", "expected_type": "agent", "expected_subtype": "string"},
    {"question": "Reverse 'abcdef'", "expected_type": "agent", "expected_subtype": "string"},
    {"question": "Uppercase 'hello'", "expected_type": "agent", "expected_subtype": "string"},
    {"question": "Max of 1, 5, 3, 9, 2", "expected_type": "agent", "expected_subtype": "statistics"},
    {"question": "Average of 10, 20, 30", "expected_type": "agent", "expected_subtype": "statistics"},
    {"question": "Sum of 1, 2, 3, 4, 5", "expected_type": "agent", "expected_subtype": "statistics"},
    {"question": "What is recursion?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain Big O notation", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is a linked list?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does quicksort work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is dynamic programming?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain binary search", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is a graph data structure?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "How does BFS work?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "What is DFS?", "expected_type": "rag", "expected_subtype": ""},
    {"question": "Explain memoization", "expected_type": "rag", "expected_subtype": ""},
]
