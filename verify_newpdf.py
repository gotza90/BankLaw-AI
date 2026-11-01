import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from banklaw_step1.query import ask

q = "\u0645\u0627 \u0647\u064a \u0634\u0631\u0648\u0637 \u0627\u0644\u062b\u0627\u0645\u0646\u0629 \u0641\u064a \u0628\u0637\u0642\u0627\u062a \u0627\u0644\u062e\u0635\u0645 \u0627\u0644\u0634\u0631\u0643\u064a\u0629 \u0639\u0644\u0649 \u0627\u0644\u0628\u0646\u0648\u0643\u061f"
result = ask(q)
print("Q:", q)
print("A:", result["answer"])
print("CITATIONS:", __import__("re").findall(r"\[Citation:.*?\]", result["answer"]))
print()
print("SOURCES:")
for item in result["context"]:
    c = item.get("citation", {})
    print("  ", c.get("document"), "|", c.get("section"))