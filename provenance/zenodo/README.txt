Raw evidence for "Matrix-free voxel elasticity on GPUs: a reproducible kernel and multigrid study across
precisions and hardware" (A. Nejeoui, A. Bekkari).

Code, protocols, verified summaries and the build that regenerates every number, table and figure of the paper:
https://github.com/nejeoui/matrix-free-voxel-elasticity (one annotated tag per experiment)

The repository holds the small files of every experiment. This record holds the large files, in two groups.
Zenodo stores files without folders, so every file name starts with its folder; the manifests map each file to
its path together with its size and SHA-256.

1. Sessions A-J (repository folder results/): MANIFEST.tsv
   session-*--voxel-session_*-evidence.tar.gz[.partNNN]
     The sealed evidence archive collected from each GPU host before the host was destroyed; its SHA-256 was
     recorded at collection time (results/<session>/voxel-*-collection.json in the repository). Unpacking an
     archive into results/<session>/remote/ restores the large state arrays (*.npz) that LARGE_FILES.tsv lists.

2. Earlier kernel and multigrid experiments (repository folders evidence/ and evidence/companion/, which hold
   their summaries): COMPANION_MANIFEST.tsv
   companion--<folder>--<rental>__evidence.tar.gz[.partNNN]
     The sealed evidence archive of each GPU experiment, unchanged, with one exception listed in REPACKED.json:
     the damping replay's archive contained a modified copy of the Traeff et al. multigrid code, which carries no
     licence and is therefore not redistributed; that archive is published without those 29 source files, and
     its original SHA-256 is recorded.
   companion--<folder>--files.tar.gz
     Every other file of the experiment folder: protocols, packages, inputs, run outputs, receipts and replays.
   EXCLUDED.tsv lists, with SHA-256 and reason, every file of these folders that is not in the files archive:
     files that are byte-identical to a member of the same folder's sealed archive (unpack the archive to
     restore them), the Traeff et al. source copies (the pinned upstream commits are referenced in the
     repository), and the files whose per-instance cloud API key was replaced by "REDACTED".

Split archives: join the parts in order before unpacking, for example
  cat session-b-20260923--voxel-session_b-evidence.tar.gz.part000 \
      session-b-20260923--voxel-session_b-evidence.tar.gz.part001 > voxel-session_b-evidence.tar.gz

Licence: CC BY 4.0 (third-party material keeps its own licence, as recorded in the repository).
