#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

def clean_helpers_file():
    """helpers.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/helpers.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ helpers.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def clean_analysis_file():
    """analysis.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/analysis.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ analysis.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def clean_nodes_file():
    """nodes.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/nodes.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ nodes.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def clean_state_file():
    """state.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/state.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ state.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def clean_timing_file():
    """timing.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/timing.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ timing.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

def clean_tools_file():
    """tools.py 파일의 잘못된 들여쓰기와 빈 줄 문제를 완전히 해결"""
    
    file_path = "parsing/parser/tarot_agent/utils/tools.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"원본 파일 크기: {len(content)} 문자")
        print(f"원본 줄 수: {len(content.splitlines())} 줄")
        
        # 정규식으로 잘못된 패턴 수정
        # 패턴 1: 함수 내부에서 빈 줄로 분리된 코드를 정상적인 들여쓰기로 변경
        
        # 1. 함수 정의 후 독스트링의 빈 줄 제거
        content = re.sub(r'(def [^:]+:)\n\n(\s+"""[^"]*""")', r'\1\n\2', content)
        
        # 2. 독스트링 후 빈 줄 제거하고 함수 본문 시작
        content = re.sub(r'(""")\n\n(\s+[^\s])', r'\1\n\2', content)
        
        # 3. 함수 내부의 모든 빈 줄 + 들여쓰기 패턴을 정상 들여쓰기로 변경
        # 패턴: \n\n    코드 → \n    코드
        content = re.sub(r'\n\n(\s{4,}[^\s])', r'\n\1', content)
        
        # 4. 함수 내부에서 연속된 빈 줄들 제거
        lines = content.split('\n')
        result_lines = []
        in_function = False
        function_indent = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # 함수 정의 감지
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                function_indent = 0
                result_lines.append(line)
                i += 1
                continue
            
            # 함수 끝 감지 (들여쓰기가 없는 줄)
            if in_function and stripped and not line.startswith(' ') and not line.startswith('\t'):
                in_function = False
            
            # 함수 내부에서 빈 줄 처리
            if in_function:
                if stripped:
                    # 첫 번째 코드 줄에서 기본 들여쓰기 설정
                    if function_indent == 0:
                        function_indent = len(line) - len(line.lstrip())
                    result_lines.append(line)
                else:
                    # 빈 줄은 건너뛰기 (함수 내부에서는 빈 줄 제거)
                    pass
            else:
                # 함수 외부에서는 빈 줄 허용 (단, 연속된 빈 줄은 1개만)
                if not stripped and result_lines and not result_lines[-1].strip():
                    pass  # 연속된 빈 줄 건너뛰기
                else:
                    result_lines.append(line)
            
            i += 1
        
        # 최종 정리
        cleaned_content = '\n'.join(result_lines)
        
        # 파일 끝 정리
        cleaned_content = cleaned_content.rstrip() + '\n'
        
        print(f"정리된 파일 크기: {len(cleaned_content)} 문자")
        print(f"정리된 줄 수: {len(cleaned_content.splitlines())} 줄")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print("✅ tools.py 파일 들여쓰기 및 빈 줄 문제 완전 해결!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    print("🧹 파일 정리 시작...")
    clean_helpers_file()
    print("\n" + "="*50 + "\n")
    clean_analysis_file()
    print("\n" + "="*50 + "\n")
    clean_nodes_file()
    print("\n" + "="*50 + "\n")
    clean_state_file()
    print("\n" + "="*50 + "\n")
    clean_timing_file()
    print("\n" + "="*50 + "\n")
    clean_tools_file()
    print("🎉 모든 파일 정리 완료!") 