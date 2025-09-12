# nerd/pipeline/tasks/tc_free.py
"""
Task for free timecourse analysis.
"""

from .base import Task

class TimecourseFreeTask(Task):
    """
    A task for performing timecourse analysis without constraints.
    """
    name = "tc_free"


    def prepare(self, cfg):
        return None, None

    def command(self, inputs, params):
        return None

    def consume_outputs(self, ctx, inputs, params, run_dir):
        pass