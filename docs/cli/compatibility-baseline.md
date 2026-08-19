# CLI compatibility baseline

Phase 0 originally recorded the command behavior present on `dev` after the durable scheduler and
ShapeMapper container work. The tests now record the Phase 2 grouped command hierarchy,
scientific workflow spellings, hidden lifecycle wrappers, database selectors, executor-profile
options, and advanced container maintenance commands without snapshotting Rich-formatted help output.

Phase 1 exposes only canonical workflow names. `probe_timecourse` replaces the removed
`probe_tc_kinetics` and `tc_free` spellings, while `nmr_kinetic_fit` consolidates degradation and
adduction fitting behind an explicit `fit_type`. Phase 2 groups durable lifecycle operations under
`nerd task`, adds `nerd run --detach`, and moves ShapeMapper maintenance under `plugin` and `image`.
The shared output contract proposed in [the CLI UX audit](../../CLI_UX_AUDIT.md) remains deferred.
