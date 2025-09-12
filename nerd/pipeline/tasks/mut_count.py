# nerd/pipeline/tasks/mut_count.py
"""
Task for mutation counting.
"""

from .base import Task

class MutCountTask(Task):
    """
    A task that calls a plugin to perform mutation counting and imports
    the resulting fmod tables.
    """
    name = "mut_count"

    def prepare(self, cfg):
        return None, None

    def command(self, inputs, params):
        return None

    def consume_outputs(self, ctx, inputs, params, run_dir):
        pass