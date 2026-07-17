
# Architecture Memory Comparison Report

## Objective

Compare CNN and Transformer memory behavior using the analytical and measured
results obtained in previous experiments.

## Methodology

The CNN experiments evaluated activation-memory scaling across different image
resolutions and batch sizes.

The Transformer experiments evaluated activation-memory scaling across
different sequence lengths and batch sizes.

## Results

CNN parameter memory is very small compared with the Transformer.

CNN memory is dominated by activation tensors generated from feature maps.

Transformer memory is strongly influenced by parameter memory and attention.

The dominant scaling variable differs between the two architectures.

## Key Findings

• CNN memory depends primarily on image resolution.

• Transformer memory depends primarily on sequence length.

• Transformer parameter memory is substantially larger.

• Different architectures require different estimation strategies.

## Conclusion

Architecture plays a fundamental role in GPU memory behavior. Memory estimators
should incorporate architecture-specific characteristics instead of relying on
a single generalized approximation.
