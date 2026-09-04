"""Errors used by Phase 0/1 stubs.

These are intentional: the skeleton reserves the call site without pretending
that inference, downloads, or training exist yet.
"""


class Phase0NotImplemented(NotImplementedError):
    """Reserved Phase 0 work item. The interface is real; the body is not."""


class Phase1NotImplemented(NotImplementedError):
    """Reserved Phase 1 work item. The interface is real; the body needs Spark / weights."""


class Phase1EvalInputMissing(FileNotFoundError):
    """A Spark-side path the in-repo scorer needs was not supplied or is absent.

    CI never requires these files. Pass ``--ablations-root``, ``--nll-jsonl``,
    ``--vap-jsonl``, and labels / transcripts when running on spark-61dd.
    """


class Phase2OutOfScope(RuntimeError):
    """Phase 2 fine-tuning is not implemented and must not start from Phase 1."""


class WeightsNotBundled(RuntimeError):
    """Raised when a caller asks this repo to locate or fetch unpublished paths.

    Public Hugging Face and GitHub ids are documented on the wrappers. This
    repository does not download them and does not invent local install paths.
    """
