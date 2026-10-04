# Terminal run callbacks

Applications can register an async callback with
`aegra_api.services.terminal_runs.set_terminal_run_callback` during their custom
FastAPI app lifespan. Pass `None` to unregister. The callback receives an immutable
`TerminalRun` after a successful terminal status commit. It is optional and does
not add tables or publication tracking.

Normal success and uncaught execution errors call it through `finalize_run`.
Worker timeouts use `execution_timeout`. Exhausted worker lease recovery uses
`lease_recovery_exhausted`. A missing execution-parameters record uses
`execution_error`. Pending retries, cancellation, shutdown requeue, and graph
interrupts do not call the application callback. The guarded final-state update
prevents a repeated finalization from invoking it again.

The snapshot reads `user_id` from the persisted run, rather than user-supplied
configuration. `cron_id` is copied from the actual cron by the scheduler and the
cron API's initial-run path. Run creation strips the reserved `aegra_cron_id`
metadata key from client input before setting it server-side. This identity is
retained inside existing `execution_params.run_metadata`, so deleting a cron does
not discard the association. Pre-extension runs and damaged execution parameters
have no authoritative cron association; callbacks must not infer one from config.

Callbacks should bound their work and catch transport errors. Callback exceptions
are logged by class name and cannot replace committed run state. Cancellation can
stop callback work. Delivery is best effort: a process crash after committing the
run can lose its notification. There is no reconciliation or replay mechanism.
