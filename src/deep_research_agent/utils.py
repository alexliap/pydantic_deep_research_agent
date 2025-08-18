import re
from datetime import datetime

import logfire


def get_date():
    return datetime.now().date()


def remove_md_headers(report: str):
    if "```markdown" in report:
        logfire.info("Removed markdown headers")
        report = report.split("```markdown")[1].split("```")[0]

    return report


def remove_generated_refs(report: str):
    regex_pattern = r"(?:#{1,3}\s*(?:\d+\.\s*)?(?:References|Sources)|\*\*(?:References|Sources):\*\*)"
    parts = re.split(regex_pattern, report)
    if len(parts) > 1:
        logfire.info("Removed generated references section")
        report_without_refs = parts[0].strip()

        return report_without_refs
    else:
        return report
