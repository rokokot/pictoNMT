# PictoNMT: Neural Machine Translation for Pictogram-to-Text Generation

PictoNMT is a neural machine translation system designed specifically for converting pictogram sequences (currently only ARASAAC) into natural language french text. This research project adapts NMT architectures with custom embedding and decoding strategies in order to address the unique challenges of cross-modal translation from pictographic to linguistic representations.

Augmentative and Alternative Communication (AAC) systems often use pictograms to help users express themselves. Converting these pictogram sequences into natural language text enables easier communication and information processing for users with intellectual disabilities or language use deficits. Rather than treating this as a bonafide multimodal(image based) task, we model it as a translation problem where pictograms serve as the source language and text as the target language. Our system is released with the following contributions:

1. Hybrid Pictogram Encoding

2. Neural Architecture for Cross-Modal Translation
We adapt the Transformer architecture specifically for pictogram-to-text translation:

Custom Encoder: Processes pictogram representations with position awareness
Enhanced Cross-Attention: Specialized attention mechanisms for handling the information asymmetry
Architecture Tuning: Optimized hyperparameters for the specific constraints of pictogram translation

3. Template Decoding
A novel decoding strategy that enhances beam search with linguistic knowledge:

Dynamic Schema Construction: Builds a linguistic blueprint from pictogram sequences
Functional Word Prediction: Leverages schema to insert grammatical elements not present in source
Schema-Guided Beam Search: Uses induced schemas to guide the generation process