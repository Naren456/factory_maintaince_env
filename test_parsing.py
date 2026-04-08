from inference import parse_action

test_cases = [
    "Repair: 0",
    "inspect 1,",
    "wait",
    "REPLACE 2!!",
    "I think we should repair machine 0",
    "unknown action 5",
    "",
    "   wait   "
]

print("Testing parse_action robustness:")
for tc in test_cases:
    action = parse_action(tc)
    print(f"Input: '{tc}' -> Parsed: type={action.type}, id={action.machine_id}")
