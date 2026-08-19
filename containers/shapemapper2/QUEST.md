# Quest SIF validation

SIF is the single-file image format used by Singularity and Apptainer. The
same digest-pinned GHCR image can be downloaded once, converted to SIF, cached
in scratch storage, and executed without Docker privileges.

Quest currently documents SingularityCE through a module such as
`singularityce/4.3.1-gcc-8.5.0`. Confirm the currently available module name at
run time. Apptainer is also supported by the validation script when its command
is available instead.

After the image has an immutable OCI digest:

```bash
module load singularityce/4.3.1-gcc-8.5.0

./containers/shapemapper2/validate-sif-on-quest.sh \
  ghcr.io/edr-choi/nerd-shapemapper2@sha256:<oci-digest> \
  "$SCRATCH/nerd/shapemapper2"
```

The script refuses mutable tags. It caches the SIF by OCI digest, runs the
upstream example through a writable bind mount, and prints the SIF SHA-256.
Store its output with the CI run URL and OCI digest.

This validation cannot be completed before an image is reachable from Quest.
For the approval-gated first release, push the image privately, authenticate
Singularity to GHCR for the test, and only then request approval to change the
package to public. Alternatively, transfer a local OCI archive to Quest for a
pre-push infrastructure check; that does not validate registry access.
