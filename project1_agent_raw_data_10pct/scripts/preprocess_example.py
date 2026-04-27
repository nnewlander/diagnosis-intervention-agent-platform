import json
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]

# 读取教师请求
dialogs = [json.loads(line) for line in open(base / 'raw' / 'raw_teacher_support_dialogs_10pct.jsonl', encoding='utf-8')]

# 读取学生画像
profiles = pd.read_csv(base / 'mysql' / 'student_profiles_10pct.csv')

# 简单示例：把请求和学生课程阶段对齐
profile_map = profiles.set_index('student_id').to_dict(orient='index')
for d in dialogs[:5]:
    sid = d.get('student_id')
    course = profile_map.get(sid, {}).get('primary_course_module')
    print(d['request_id'], d['raw_request'], '=>', course)
