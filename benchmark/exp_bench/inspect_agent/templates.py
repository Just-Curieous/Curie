from string import Template

additional_notes_template = Template(
    """
ADDITIONAL NOTES
------
- **Compute**: You have access to a ${type_of_processor} with the appropriate drivers installed.
- **Total Runtime**: You are expected to work until you have spent ${max_time_in_hours} hours on this task. You may iteratively develop your solution within these constraints, but bear in mind that program runtime also counts towards the time limit. This is a difficult task and we expect a good solution to require using the full time limit.
- **API keys**: LLM related credentials (if needed) are available in /workspace/setup_apis_exp/
- The code repo is available in /workspace. Make changes only within this directory.
- You have root access in your environment.
- You must not stop until you've completed the experiment tasks.
- Remember, you must actually create and perform/conduct the experiment tasks, not just write a plan for how to do so or mock results.
"""
)
