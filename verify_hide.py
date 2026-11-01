import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import re
import streamlit.testing.v1 as st_test
from banklaw_step1.query import ask

result = ask("\u0645\u0627 \u0647\u064a \u0645\u062f\u0629 \u0627\u0644\u0631\u062f \u0639\u0644\u0649 \u0634\u0643\u0648\u0629 \u0627\u0644\u0639\u0645\u0644\u0627\u0621\u061f")

import banklaw_ui

at = st_test.AppTest.from_function(
    banklaw_ui.build_app,
    kwargs={"ask": lambda q: result},
    default_timeout=120,
)
at.run()

chat_inputs = [el for el in at if el.type == "chat_input"]
chat_inputs[0].set_value("test question").run()

markdown = "\n".join(el.value for el in at if el.type == "markdown")
cites = re.findall(r"\[Citation:.*?\]", markdown)
print("VISIBLE CITATIONS:", len(cites))
print("HAS CITATION TAG:", "[Citation:" in markdown)
print("ANSWER PREVIEW:", result["answer"][:120])