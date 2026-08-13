Test input placeholder
======================

The large T1w test volume that used to live at

    test/input/interf0/images/t1-brain-mri/input.mha

was stripped from this public release (heavy binary fixture,
> 5 MB). To reconstruct a working test input, stage a real T1w
scan into the same location by running:

    bash ../stage_test_input.sh <session-id>

That helper reads a T1w NIfTI from a local RAW BIDS tree (whose
absolute path you must set inside stage_test_input.sh - see the
comment on the RAW_ROOT variable) and writes it back here as
`input.mha` (SimpleITK-compressed), exactly the on-portal format
Grand Challenge feeds the container.

The interface-manifest files at

    interf0/inputs.json
    interf0/stroke-metadata.json

are kept as-is and describe the socket layout the container
expects at runtime.

Reference outputs
-----------------
`../output/interf0/images/stroke-lesion-segmentation/output.mha`
(uint8 lesion mask) is retained because it is small (~125 KB).
The corresponding lesion-probability-map/output.mha was stripped
for size and will be regenerated when you run the container on a
freshly staged input.
