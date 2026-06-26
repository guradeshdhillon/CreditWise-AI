import json

log_file = r"C:\Users\gurad\.gemini\antigravity-ide\brain\49e39f5f-d36c-431a-88a5-579be24764a4\.system_generated\logs\transcript.jsonl"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            step = json.loads(line)
            content = step.get("content", "")
            if not content:
                content = str(step.get("thinking", ""))
            
            # check tool calls as well
            tool_calls = step.get("tool_calls", [])
            for tc in tool_calls:
                content += " " + str(tc.get("args", ""))
            
            if "bottom" in content.lower():
                print(f"Step {step.get('step_index')} ({step.get('type')}, {step.get('source')}):")
                # print first 500 characters of content/thinking
                print(content[:600])
                print("-" * 50)
        except Exception as e:
            print("Error parsing line:", e)
