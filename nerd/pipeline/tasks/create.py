# nerd/pipeline/tasks/create.py
"""
Task for creating and ingesting initial data.
"""

from .base import Task

class CreateTask(Task):
    """
    A task for initial data ingestion, either from inline configuration
    or from external sheets.
    """
    name = "create"

    def prepare(self, cfg):
        return None, None

    def command(self, inputs, params):
        return None

    def consume_outputs(self, ctx, inputs, params, run_dir):
        pass
