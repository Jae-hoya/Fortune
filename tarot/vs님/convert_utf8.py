import os

def convert_to_utf8(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content.decode('utf-16')
        except UnicodeDecodeError:
            text = content.decode('cp949', errors='replace')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"{filepath} → UTF-8 변환 완료")

# parsing 폴더에서 실행할 때는 아래처럼!
# convert_to_utf8('parser/tarot_agent/agent.py')
# convert_to_utf8('parser/tarot_agent/utils/analysis.py')
# convert_to_utf8('parser/tarot_agent/utils/helpers.py')
# convert_to_utf8('parser/tarot_agent/utils/nodes.py')
# convert_to_utf8('parser/tarot_agent/utils/state.py')
# convert_to_utf8('parser/tarot_agent/utils/timing.py')
# convert_to_utf8('parser/tarot_agent/utils/tools.py')
# convert_to_utf8('parser/tarot_agent/utils/translation.py')
# convert_to_utf8('parser/tarot_agent/utils/web_search.py')
convert_to_utf8('parser/tarot_agent/__init__.py')
# convert_to_utf8('parser/__init__.py')
# convert_to_utf8('__init__.py')
# convert_to_utf8('parser/tarot_agent/utils/__init__.py')