#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

print("🔍 시작...")

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"📁 현재 경로: {os.getcwd()}")
print(f"🐍 Python 경로: {sys.path[:3]}")

try:
    print("📦 parsing 모듈 import 시도...")
    import parsing
    print("✅ parsing 모듈 import 성공")
    
    print("📦 parsing.parser 모듈 import 시도...")
    import parsing.parser
    print("✅ parsing.parser 모듈 import 성공")
    
    print("📦 parsing.parser.tarot_agent 모듈 import 시도...")
    import parsing.parser.tarot_agent
    print("✅ parsing.parser.tarot_agent 모듈 import 성공")
    
    print("📦 parsing.parser.tarot_agent.agent 모듈 import 시도...")
    import parsing.parser.tarot_agent.agent
    print("✅ parsing.parser.tarot_agent.agent 모듈 import 성공")
    
    print("🚀 main 함수 실행 시도...")
    from parsing.parser.tarot_agent.agent import main
    main()
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc() 